import streamlit as st
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import os
import numpy as np
import torch
import openai
import re
from apiclient import discovery
from httplib2 import Http
from oauth2client import client, file, tools

st.title("AudiQuiz Generator")

model = whisper.load_model("base")

def record(duration):
    fs = 44100
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    st.write("Recording Audio - Speak now!")
    sd.wait()
    #st.write("Audio Recording Complete")
    write('output.mp3', fs, myrecording)

def transcribe():
    torch.cuda.is_available()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base", device=DEVICE)
    

    audio = whisper.load_audio("output.mp3")
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    _, probs = model.detect_language(mel)
    #st.write(f"Detected language: {max(probs, key=probs.get)}")

    options = whisper.DecodingOptions(language='en', without_timestamps=True, fp16=False)
    result = whisper.decode(model, mel, options)
    #st.write(result.text)

    result = model.transcribe("output.mp3")
    #st.write(result["text"])
    return result["text"]

def analyser_resultat(quiz):
    questions = []
    options = []
    sous_options = []
    reponses = []

    pattern_question = re.compile(r'^\d+(?:\)|\.|\-)(.+\?$)')
    pattern_option = re.compile(r'^[a-zA-Z](?:\)|\.|\-)(.+$)')
    pattern_reponse = re.compile(r'Answer:\s[a-zA-Z](?:\)|\.|\-)(.+$)')

    for ligne in quiz.splitlines():
        ligne = ligne.strip()

        if ligne == '':
            if sous_options:
                options.append(sous_options)
                sous_options = []
        else:
            if pattern_question.match(ligne):
                questions.append(ligne)
            if pattern_option.match(ligne):
                sous_options.append(ligne)
            if pattern_reponse.match(ligne):
                reponses.append(ligne)

    if sous_options:
        options.append(sous_options)

    return questions, options

def getGpt(text):
    openai.api_key = "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"
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Generate a 10 question quiz with 3 choices about {text}"}
        ]
    )

    #st.write(completion.choices[0].message.content)

    return completion.choices[0].message.content

def generate_quiz(questions, options):
    SCOPES = "https://www.googleapis.com/auth/forms.body"
    DISCOVERY_DOC = "https://forms.googleapis.com/$discovery/rest?version=v1"

    store = file.Storage('token.json')
    creds = None
    if not creds or creds.invalid:
        flow = client.flow_from_clientsecrets('client_secret_845279617265-suhs30ghi4qf3v8sqrrd41gdsq6m8mo5.apps.googleusercontent.com.json', SCOPES)
        creds = tools.run_flow(flow, store)

    form_service = discovery.build('forms', 'v1', http=creds.authorize(Http()), discoveryServiceUrl=DISCOVERY_DOC, static_discovery=False)

    NEW_FORM = {
        "info": {
            "title": "Quiz",
        }
    }
    result = form_service.forms().create(body=NEW_FORM).execute()

    update = {
        "requests": [
            {
                "updateSettings": {
                    "settings": {
                        "quizSettings": {
                            "isQuiz": True
                        }
                    },
                    "updateMask": "quizSettings.isQuiz"
                }
            }
        ]
    }
    question_setting = form_service.forms().batchUpdate(formId=result["formId"], body=update).execute()

    for i in range(len(questions)):
        NEW_QUESTION = {
            "requests": [
                {
                    "createItem": {
                        "item": {
                            "title": questions[i],
                            "questionItem": {
                                "question": {
                                    "required": True,
                                    "choiceQuestion": {
                                        "type": "RADIO",
                                        "options": [{"value": j} for j in options[i]],
                                        "shuffle": True
                                    }
                                }
                            },
                        },
                        "location": {
                            "index": i
                        }
                    }
                }
            ]
        }
        question_setting = form_service.forms().batchUpdate(formId=result["formId"], body=NEW_QUESTION).execute()

    get_result = form_service.forms().get(formId=result["formId"]).execute()
    return get_result['responderUri']

def main():
    duration = st.number_input("Recording duration (in seconds)", value=5, min_value=1)
    if st.button("Record Audio"):
        record(duration)
        st.write("Audio recording complete!")

        if os.path.exists("output.mp3"):
            st.audio("output.mp3", format="audio/mp3")
            loading_message = st.empty()
            loading_message.text("Transcribing audio...")

            #st.write("Transcribing audio...")
            text = transcribe()
             # Display loading message
            
            loading_message.text("Quiz is loading...")
            res = getGpt(text)

            questions, options = analyser_resultat(res)
            url = generate_quiz(questions, options)
            # Update loading message
            loading_message.text("Quiz generated successfully!")
            #st.success("Quiz generated successfully!")
            st.write("Go to the generated quiz:")
            st.markdown(f"[Click here to go to the Quiz]({url})", unsafe_allow_html=True)

if __name__ == "__main__":
    main()