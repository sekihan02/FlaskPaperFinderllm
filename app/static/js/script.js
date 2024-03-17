document.addEventListener('DOMContentLoaded', (event) => {
    var socket = io();
    var selectedFile = null; // 選択されたファイルを保持する変数

    // ファイル選択時の処理
    document.getElementById('fileInput').addEventListener('change', function(e) {
        handleFileSelection(e.target.files);
    });

    // ドラッグアンドドロップ時の処理
    var dropArea = document.getElementById('dropArea');
    dropArea.addEventListener('dragover', function(e) {
        e.preventDefault();
    });
    dropArea.addEventListener('drop', function(e) {
        e.preventDefault();
        handleFileSelection(e.dataTransfer.files);
    });

    function handleFileSelection(files) {
        if (files.length > 0) {
            selectedFile = files[0]; // 選択されたファイルを保持
            document.getElementById('fileInfo').innerText = `ファイルを読み込みました: ${selectedFile.name}`;
        }
    }

    // ファイルアップロード処理の追加
    function uploadFile(file) {
        var formData = new FormData();
        formData.append('file', file);

        fetch('/upload', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            console.log(data.message);
            // アップロード完了後の処理（例: 応答メッセージの表示）
        })
        .catch(error => {
            console.error('アップロード中にエラーが発生しました:', error);
        });
    }
    // フォームの送信処理
    $('#chatForm').submit(function(e) {
        e.preventDefault();
        var message = $('#userInput').val().trim();

        // ファイルが選択されていれば、その情報をサーバーに送信
        if (selectedFile) {
            uploadFile(selectedFile);
            selectedFile = null; // ファイル送信後は選択されたファイルをリセット
            document.getElementById('fileInfo').innerText = ''; // ファイル情報の表示をクリア
        }
        else if (message) {
            showLoadingScreen(); // メッセージ送信前にローディング画面を表示
            socket.emit('send_message', { message: message });
            // ユーザーのメッセージに含まれる改行を<br>タグに置換して表示
            var formattedMessage = message.replace(/\n/g, '<br>');
            $('#chat-box').append(`<div class="user-message">User: ${formattedMessage}</div>`);
            $('#userInput').val('');
            // メッセージボックスを自動スクロール
            var chatcontainer = document.getElementById('chat-container');
            chatcontainer.scrollTop = chatcontainer.scrollHeight;
        }
    });

    socket.on('receive_message', function(data) {
        hideLoadingScreen(); // メッセージ受信後にローディング画面を非表示に
        // ボットのメッセージに含まれる改行を<br>タグに置換して表示
        var formattedMessage = data.message.replace(/\n/g, '<br>');
        $('#chat-box').append(`<div class="bot-message">Bot: ${formattedMessage}</div>`);
        // メッセージボックスを自動スクロール
        var chatcontainer = document.getElementById('chat-container');
        chatcontainer.scrollTop = chatcontainer.scrollHeight;
    });

    $('#userInput').keydown(function(e) {
        if (e.ctrlKey && e.keyCode === 13) {
            $('#chatForm').submit();
            e.preventDefault();
        }
    });
});


function showLoadingScreen() {
    // ロード画面を表示するコード
    const loadingScreen = document.createElement('div');
    loadingScreen.id = 'loadingScreen';
    // loadingScreen.innerHTML = '<div class="loader">Loading...</div>';
    loadingScreen.innerHTML = '<div class="loader"></div>';
    document.body.appendChild(loadingScreen);
}

function hideLoadingScreen() {
    // ロード画面を非表示にするコード
    const loadingScreen = document.getElementById('loadingScreen');
    if (loadingScreen) {
        loadingScreen.remove();
    }
}

