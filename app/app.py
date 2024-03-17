from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import os
import json
import datetime as dt
import warnings
import pdfplumber
import re
import arxiv
import openai
from openai import OpenAI
from dotenv import load_dotenv
from duckduckgo_search import DDGS, AsyncDDGS
import asyncio

from time import time

class Timer:
    def __init__(self, logger=None, format_str="{:.3f}[s]", prefix=None, suffix=None, sep=" "):

        if prefix: format_str = str(prefix) + sep + format_str
        if suffix: format_str = format_str + sep + str(suffix)
        self.format_str = format_str
        self.logger = logger
        self.start = None
        self.end = None

    @property
    def duration(self):
        if self.end is None:
            return 0
        return self.end - self.start

    def __enter__(self):
        self.start = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time()
        out_str = self.format_str.format(self.duration)
        if self.logger:
            self.logger.info(out_str)
        else:
            print(out_str)
# すべての警告を無視する
warnings.filterwarnings('ignore')

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['UPLOAD_FOLDER'] = 'uploads'  # アップロードされたファイルを保存するフォルダ
socketio = SocketIO(app)

openai.api_key = os.getenv("OPENAI_API_KEY")
# MODEL_NAME = "gpt-3.5-turbo-0125"
# MODEL_NAME = "gpt-3.5-turbo-instruct"
MODEL_NAME = "gpt-4-0125-preview"

TEMPERATURE = 0.7

class StreamingLLMMemory:
    """
    StreamingLLMMemory クラスは、最新のメッセージと特定数のattention sinksを
    メモリに保持するためのクラスです。
    
    attention sinksは、言語モデルが常に注意を向けるべき初期のトークンで、
    モデルが過去の情報を"覚えて"いるのを手助けします。
    """
    def __init__(self, max_length=10, attention_sinks=4):
        """
        メモリの最大長と保持するattention sinksの数を設定
        
        :param max_length: int, メモリが保持するメッセージの最大数
        :param attention_sinks: int, 常にメモリに保持される初期トークンの数
        """
        self.memory = []
        self.max_length = max_length
        self.attention_sinks = attention_sinks
    
    def get(self):
        """
        現在のメモリの内容を返します。
        
        :return: list, メモリに保持されているメッセージ
        """
        return self.memory
    
    def add(self, message):
        """
        新しいメッセージをメモリに追加し、メモリがmax_lengthを超えないように
        調整します。もしmax_lengthを超える場合、attention_sinksと最新のメッセージを
        保持します。
        
        :param message: str, メモリに追加するメッセージ
        """
        self.memory.append(message)
        if len(self.memory) > self.max_length:
            self.memory = self.memory[:self.attention_sinks] + self.memory[-(self.max_length-self.attention_sinks):]
    
    def add_pair(self, user_message, ai_message):
        """
        ユーザーとAIからのメッセージのペアをメモリに追加します。
        
        :param user_message: str, ユーザーからのメッセージ
        :param ai_message: str, AIからのメッセージ
        """
        # self.add("User: " + user_message)
        # self.add("AI: " + ai_message)
        self.add({"role": "user", "content": user_message})
        self.add({"role": "assistant", "content": ai_message})
    
    # ここにはStreamingLLMとのインタラクションのための追加のメソッドを
    # 実装することもできます。例えば、generate_response, update_llm_modelなどです。

# 16件のメッセージを記憶するように設定
memory = StreamingLLMMemory(max_length=16)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('send_message')
def handle_message(data):
    user_message = data['message']
    repor_checked = data.get('repor', False)  # チェックボックスがチェックされている場合はTrue、そうでない場合はFalse

    # チェックボックスの状態に基づいて何らかの処理を行う
    if repor_checked:
        print("Repor is checked.")
        # Reporがチェックされている場合の処理をここに追加

        report_message = "Repor checked"
        emit('receive_message', {'message': report_message})
    else:
        print("Repor is not checked.")
        # Reporがチェックされていない場合の処理をここに追加
        # ユーザーからのメッセージに応じて、ボットの応答を送信
        num_questions = 5
        # OpenAIクライアントの初期化
        client = OpenAI()
        questions_and_purposes = generate_research_questions_and_purpose_with_gpt(user_message, num_questions, client)
        generate_search_string = generate_search_string_with_gpt(user_message, questions_and_purposes, client)
        emit('receive_message', {'message': "\n検索クエリ生成結果\n"+ str(generate_search_string)})
    
        simplified_queries = simplify_search_queries(generate_search_string)
        # デフォルトのAPIクライアントを構築する。
        arxivclient = arxiv.Client()
        for query in simplified_queries:
            search = arxiv.Search(
                query = query,
                max_results = 3,
                sort_by = arxiv.SortCriterion.SubmittedDate
            )

            # 検索を実行し、結果を取得する。
            results = arxivclient.results(search)
            # 取得した論文のタイトルを1件ずつ表示する。
            for r in results:
                emit('receive_message', {'message': f"\n{str(r.title)}\n{get_summary(r, client)}\n{r}"})
    # # 会話履歴の保存
    # memory.add_pair(user_message, res_message)
    # # 以前の会話をメモリから取得
    # past_conversation = memory.get()
    # emit('receive_message', {'message': str(past_conversation)})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        # 全ての接続されたクライアントに対してメッセージを送信
        socketio.emit('receive_message', {'message': f"ファイルは受けとってません"})
        
        return 'No selected file'
    # if file:
    #     filename = secure_filename(file.filename)
    #     save_path = os.path.join('uploads', filename)
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     file.save(save_path)

    #     # 全ての接続されたクライアントに対してメッセージを送信
    #     socketio.emit('receive_message', {'message': f"{filename} を受け取りました。"})
        
    #     return 'File uploaded successfully'
    
    # ファイルを読み込んだ場合
    with Timer(prefix="pdf_to_text"):
        paper_text = pdf_to_text(file)
        # 全ての接続されたクライアントに対してメッセージを送信
        # socketio.emit('receive_message', {'message': f"ファイルをテキストに変換しました。"})
        # socketio.emit('receive_message', {'message': f"{paper_text}"})
        
    # OpenAIクライアントの初期化
    client = OpenAI()
    # 概要と提案手法名抽出の実行
    with Timer(prefix="summary_method_name"):
        summary_method_name = extract_summary_method_name(MODEL_NAME, paper_text, client)

        # ステップ1: summaryの型を確認
        summary = summary_method_name['summary']
        if isinstance(summary, dict):
            # ステップ2: 辞書型の場合、値を抽出し整形
            formatted_summary = '\n'.join([str(value) for value in summary.values()])
        else:
            # 文字列型の場合、そのまま使用
            formatted_summary = summary
        
        # ステップ3: 整形した文字列を送信
        socketio.emit('receive_message', {'message': f"\n概要: {formatted_summary}"})


    with Timer(prefix="explain_method_algorithm"):
        method_algorithm = explain_method_algorithm(MODEL_NAME, paper_text, str(summary_method_name["method_name"]), client)
        
        # ステップ1: method_algorithmの型を確認
        method_alg = method_algorithm['method']
        if isinstance(method_alg, dict):
            # ステップ2: 辞書型の場合、値を抽出し整形
            formatted_method = '\n'.join([str(value) for value in method_alg.values()])
        else:
            # 文字列型の場合、そのまま使用
            formatted_method = method_alg
        
        # ステップ3: 整形した文字列を送信
        socketio.emit('receive_message', {'message': f"\n使用手法:{str(summary_method_name['method_name'])}\n手法の説明:\n{formatted_method}"})
    
    with Timer(prefix="generate_pseudocode_for_method"):
        code_method = generate_pseudocode_for_method(MODEL_NAME, str(formatted_method), client)
        
        socketio.emit('receive_message', {'message': f"\nアルゴリズム:\n{code_method['code']}"})

    return 'File uploaded successfully'

if __name__ == '__main__':
    socketio.run(app, debug=True)


# PDFファイルを読み込み、テキストに変換する関数
def pdf_to_text(file_path):
    with pdfplumber.open(file_path) as pdf:
        return ''.join(page.extract_text() for page in pdf.pages)


# 取り込んだファイルに対して実行する
def extract_summary_method_name(model_name, text, client):
    # 概要と提案手法名を生成する
    prompt = [{'role': 'system', 'content': "Please compile the following text into a research report so that it can be fully understood without any omissions. If a method is proposed, extract its name."}]
    prompt.append({"role" : "system", "content" : 'Please generate the research report to include the following contents.\n"CHALLENGE" The current situation faced by the researcher; it will normally include a Problem Statement, the Motivation, a Hypothesis and/or a Goal.\n"APPROACH" How they intend to carry out the investigation, comments on a theoretical model or framework.\n"OUTCOME" Overall conclusion that should reject or support the research hypothesis.'})
    
    prompt.append({"role" : "system", "content" : 'Please create a research report by referencing the example research report below, structuring it around three distinct aspects in order: "Challenge," "Approach," and "Outcome."\n"Challenge" Existing statistical phrasal orhierarchical machine translation system relies on a large set of translation rules which results in engineering challenges. \n"Approach": The proposed method consistently outperforms existing methods in BLEU on various low-resource language translation tasks with less training data.\n"Outcome" They propose to use factorized grammar from the field of linguistics as more general translation rules from XTAG English Grammar.'})
    
    prompt.append({"role": "system", "content": "Results must be in Japanese."})
    prompt.append({"role" : "system", "content" : "Outputs should be generated in step by step."})
    prompt.append({"role": "system", "content": "Think the option as hypothesis. Whether it entails with those premises?"})
    prompt.append({"role" : "system", "content" : "Please explain the research report in a way that is easy to understand for high school students, without making it complicated. It's okay if the explanation becomes lengthy."})
    
    prompt.append({"role": "system", "content": "Please format the output in Markdown."})
    prompt.append({"role": "system", "content": "Results must be in Japanese."})
    
    prompt.append({"role": "user", "content": 'Based on the input text, generate a JSON containing two different pieces of information. First, use "extract_summary" as the schema to create a section that keys in the result of summarizing the content of the text accurately and completely. Next, use "method_name" as the schema to create a section that keys in the name of the method proposed or used within the text. Combine these pieces of information to output as a single JSON object. This JSON will be formatted as {"extract_summary": [the result of summarizing the text content], "method_name": [the name of the method proposed or used within the text]}.'})
    
    prompt.append({"role": "user", "content": f"Input text: {text}"})
    prompt.append({"role": "user", "content": f"Include:\n- Overview\n- Novelty\n- Methodology\n- Results"})
    """
    システム
    あなたは以下の text を過不足なく理解できるように調査報告書としてまとめ、提案されている手法がある場合はその名称を抽出してください。
    調査報告書は以下の内容を含む形で生成してください\n
    • CHALLENGE: The current situation faced by the researcher; it will normally include a Problem Statement, the Motivation, a Hypothesis and/or a Goal.\n
    • APPROACH: How they intend to carry out the investigation, comments on a theoretical model or framework.\n
    • OUTCOME: Overall conclusion that should reject or support the research hypothesis.

    調査報告書は以下の例を参考に「課題」「アプローチ」「結果」の3つの異なる側面を順番に作成してください。概要: [ACLSum: A New Dataset for Aspect-based Summarization of Scientific Publications](https://arxiv.org/abs/2403.05303v1)のデータセット例を一つ
    出力は Markdown 形式にしてください。
    
    結果は日本語でなければならない。
    user
    入力されたテキストに基づき、二つの異なる情報を含むJSONを生成します。最初に、"extract_summary"をスキーマとして使用し、テキストの内容を過不足なく概要としてまとめた結果をキーとする部分を生成します。次に、"method_name"をスキーマとして使用し、テキスト内で提案されたり使用されている手法名をキーとする部分を生成します。これらの情報を組み合わせて、一つのJSONオブジェクトとして出力します。このJSONは、{"extract_summary": [テキストの内容を概要としてまとめた結果], "method_name": [テキスト内で提案されたり使用されている手法名]}の形式で表されます。
    
    入力されたテキスト: {text}
    """
    
    # 概要と提案手法名抽出用のプロンプトテンプレートを作成
    response = client.chat.completions.create(
        model=model_name, # model = "deployment_name".
        messages=prompt,
        response_format={ "type": "json_object" },
        temperature=TEMPERATURE,
    )
    summary_method_name_str = response.choices[0].message.content
    # print(summary_method_name_str)
    
    # JSON形式の文字列を辞書に変換
    summary_method_name = json.loads(summary_method_name_str)
    
    # 出力と新しいメッセージをステートに反映
    return {
        "summary": summary_method_name["extract_summary"],
        "method_name": summary_method_name["method_name"],
    }


def explain_method_algorithm(model_name, text, method_name, client):
    # アルゴリズムの説明
    prompt = [{'role': 'system', 'content': "Please explain the algorithm of the method name from the following text in detail, using both sentences and formulas. Carefully describe the mechanism in order, ensuring that it can be understood and implemented. Design the process flow in a way that allows for the algorithm to be implemented without any omissions or excess."}]
    prompt.append({"role" : "system", "content" : "Describe the algorithm in detail, explaining what it aims to achieve, how it processes to accomplish this, and how exactly these processes are carried out, regardless of the length of the explanation. Just ensure it is accurate."})
    prompt.append({"role" : "system", "content" : "Outputs should be generated in step by step."})
    prompt.append({"role" : "system", "content" : "Please format the output in Markdown."})
    prompt.append({"role": "system", "content": "Results must be in Japanese."})
    prompt.append({"role": "system", "content": 'Please generate a JSON from the following input text. Use "method" as the schema, and for the key, use "the detailed explanation of the processing of the method_name algorithm in simple language". Generate it in the format of {"method": "the result of a detailed explanation of the method_names algorithm described in simple language"}.'})
    
    prompt.append({"role": "user", "content": 'Generate a JSON from the following input text. Use "method" as the schema, and use the judgment result as the key, to create it in the format {"method": the result of grouping the search_query based on relevance into a list format that can be used in Python}.'})
    
    prompt.append({"role": "user", "content": f"Input text: {text}"})
    prompt.append({"role": "user", "content": f"method name: {method_name}"})
    
    """
    システム
    あなたは以下の text から method name のアルゴリズムを順番に過不足なく文章と数式で丁寧に順番に仕組みが理解でき、実装をするための処理の流れを設計できるように説明してください。
    アルゴリズムの説明は、何を実現するために、どのように処理を実行し、その処理はどのように実行されるのかをどれだけ長くなってもよいのでとにかく正確に説明してください。
    出力は Markdown 形式にしてください。
    
    結果は日本語でなければならない。
    以下の入力テキストからJSONを生成してください。スキーマには "method"、キーには"text から method_name のアルゴリズムを平易な文章で処理内容を詳細に説明した内容"を使ってください。"method": "method_name のアルゴリズムを平易な文章で処理内容を詳細に説明した内容した結果}'}の形式で生成してください。
    user
    
    入力されたテキスト: {text}
    method name: {method_name}
    """
    
    # 概要と提案手法名抽出用のプロンプトテンプレートを作成
    method = client.chat.completions.create(
        model=model_name, # model = "deployment_name".
        messages=prompt,
        response_format={ "type": "json_object" },
        temperature=TEMPERATURE,
    )
    method_str = method.choices[0].message.content
    
    # JSON形式の文字列を辞書に変換
    method_algorithm = json.loads(method_str)
    
    # 出力と新しいメッセージをステートに反映
    return {
        "method": method_algorithm["method"],
    }


def generate_pseudocode_for_method(model_name, algorithm, client):
    # アルゴリズムの説明
    prompt = [{'role': 'system', 'content': "Based on the description of the following algorithm, please create a comprehensive pseudo-implementation code in Python without omitting any details."}]
    prompt.append({"role" : "system", "content" : "Outputs should be generated in step by step."})
    prompt.append({"role": "system", "content": "Please format the output in Markdown."})
    prompt.append({"role": "system", "content": "Comment must be in Japanese."})
    prompt.append({"role": "system", "content": 'Please generate a JSON from the following input text. Use "code" as the schema, and for the key, use "the result of generating code that executes the algorithm of algorithm in Python". Generate it in the format of {"code": "the result of reproducing the algorithm algorithm in Python code"}.'})
        
    prompt.append({"role": "user", "content": f"algorithm: {algorithm}"})
    
    """
    システム
    あなたは以下の algorithm の説明を基にpythonの疑似実装コードを過不足なく作成してください。
    出力は Markdown 形式にしてください。
    
    コメントは日本語でなければならない。
    以下の入力テキストからJSONを生成してください。スキーマには "code"、キーには"algorithm のアルゴリズムをpythonのコードで動くようにコードを生成した結果"を使ってください。"code": "algorithm のアルゴリズムをpythonのコードで再現した結果"}'}の形式で生成してください。
    user
    
    algorithm: {algorithm}
    """
    
    # 概要と提案手法名抽出用のプロンプトテンプレートを作成
    code_res = client.chat.completions.create(
        model=model_name, # model = "deployment_name".
        messages=prompt,
        response_format={ "type": "json_object" },
        temperature=TEMPERATURE,
    )
    code_str = code_res.choices[0].message.content
    print(code_str)
    
    # JSON形式の文字列を辞書に変換
    code = json.loads(code_str)
    
    # 出力と新しいメッセージをステートに反映
    return {
        "code": code["code"],
    }



def generate_research_questions_and_purpose_with_gpt(objective, num_questions, client):
    # プランナーエージェント: 研究目的から研究質問と検索文字列を生成します
    # Construct the prompt dynamically
    prompt_content = f"You are a helpful assistant capable of generating research questions along with their purposes for a systematic literature review.\n"
    prompt_content = f"Given the research objective: '{objective}', generate {num_questions} distinct research questions, each followed by its specific purpose. 'To examine', or 'To investigate'."
    
    response = client.chat.completions.create(
        # model="gpt-3.5-turbo",
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "You are a helpful assistant capable of generating research questions along with their purposes for a systematic literature review."},
            {"role": "user", "content": prompt_content}
        ],
        # response_format={ "type": "json_object" },
        temperature=TEMPERATURE,
    )
    result = response.choices[0].message.content
    return {"research_questions": result}


def extract_search_strings(content):
    possible_operators = ['AND', 'OR', 'NOT', '"']
    search_strings = []
    for line in content.split('\n'):
        if any(op in line for op in possible_operators):
            search_strings.append(line.strip())  # strip()を追加して余分な空白を削除
    return search_strings if search_strings else [content]

def generate_search_string_with_gpt(objective, research_questions, client):
    # 生成された検索文字列を使用して学術データベースをクエリし、関連論文の初期セットを取得します。
    # Removed the explicit instruction for logical operators
    combined_prompt = f"Given the research objective: '{objective}', and the following research questions: {research_questions['research_questions']}, generate two concise search string for identifying relevant literature for literature review.Do not include OR. Use AND if needed."

    response = client.chat.completions.create(
        # model="gpt-3.5-turbo",
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": combined_prompt}
        ],
        # response_format={ "type": "json_object" },
        temperature=TEMPERATURE,
    )
    
    content = response.choices[0].message.content
    search_string = extract_search_strings(content)
    return search_string


# 論文の要約を取得する関数
def get_summary(result, client):
    SYSTEM = """
    ### 指示 ###
    論文の内容を理解した上で，重要なポイントを箇条書きで3点書いてください。

    ### 箇条書きの制約 ###
    - 最大3個
    - 日本語
    - 箇条書き1個を50文字以内

    ### 対象とする論文の内容 ###
    {text}

    ### 出力形式 ###
    タイトル(和名)

    - 箇条書き1
    - 箇条書き2
    - 箇条書き3
    """
    text = f"title: {result.title}\nbody: {result.summary}"

    messages = [
        {"role" : "system", "content" : SYSTEM},
        {"role": "user", "content": text}
    ]
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content


def simplify_search_queries(complex_queries):
    simplified_queries = []

    for query in complex_queries:
        # 数字とピリオドを除去して、クエリの本体だけを抽出
        clean_query = re.sub(r'^\d+\.\s*', '', query)
        
        # 括弧を除去
        clean_query = re.sub(r'[()"]', '', clean_query)
        
        # 'AND' と 'OR' で分割
        split_queries = re.split(r'\sAND\s|\sOR\s', clean_query)
        
        # 分割したクエリをリストに追加
        for sub_query in split_queries:
            sub_query = sub_query.strip()
            if sub_query and sub_query not in simplified_queries:
                simplified_queries.append(sub_query)
                
    return simplified_queries