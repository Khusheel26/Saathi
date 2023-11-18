from flask import Flask, render_template, jsonify, request
import chatbot
import depression_training

app = Flask(__name__, static_folder='static')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def chatbot_response():
        if request.method == 'POST':
            user_input = request.form.get('user_input')
            response = chatbot.chatbot_response(user_input)
            return jsonify({"response": response})           

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True)