import json
import random
from difflib import get_close_matches

def load_knowledge_base(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_knowledge_base(file_path: str, data: dict):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def find_best_matches(user_question: str, questions: list[str]) -> str | None:
    matches: list = get_close_matches(user_question, questions, n=1, cutoff=0.9)
    return matches[0] if matches else None

def get_random_answer_for_question(question_data: dict) -> str | None:
    answers = question_data.get('answers')
    return random.choice(answers) if answers else None

def training():
    knowledge_base = load_knowledge_base('knowledge_base.json')
    
    while True:
        user_input = input('You:')
        
        if user_input.lower() == 'exit':
            break
        
        best_match = find_best_matches(user_input, [q['question'] for q in knowledge_base['questions']])
        
        if best_match:
            question_data = next(q for q in knowledge_base['questions'] if q['question'] == best_match)
            answers = question_data.get('answers')
            if answers:
                answer = random.choice(answers)
                print(f'Saathi: {answer}')
            else:
                print('Bot: I have no answers for that question.')
        else:
            print('Bot: Sorry, I do not understand.')
            print('Bot: Please provide me with an answer.')
            user_answer = input('Type the answer or type "exit" to quit:')
            
            if user_answer.lower() != 'exit':
                knowledge_base["questions"].append({'question': user_input, 'answers': [user_answer]})
                save_knowledge_base('knowledge_base.json', knowledge_base)
                print('Bot: Thank you for teaching me.')

if __name__ == '__main__':
    training()
