
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="My Deary", page_icon="🐧", layout="centered")

idioma = st.selectbox("🌐 Choose your language / Escolha seu idioma", ["Português 🇧🇷", "English 🇺🇸"])
lang = "pt" if "Português" in idioma else "en"

if lang == 'pt':
    st.info("ℹ️ Para uma melhor experiência, recomendamos usar em inglês.")

with open("styles/theme.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.image("assets/mascot.png", width=120, caption="Oi! Eu sou o Deary 🐧" if lang == "pt" else "Hi! I'm Deary 🐧")

st.title("Um amigo para seus desabafos 💗" if lang == "pt" else "A friend for your thoughts 💗")
st.markdown("_Versão em testes – aqui pra ouvir você com carinho._" if lang == "pt" else "_Beta version – here to listen to you with care._")

MODEL_NAME = "cardiffnlp/twitter-roberta-base-emotion"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

respostas_emocionais = {
    "sadness": {
        "reflexao": {
            "pt": "Tudo que pesa agora não precisa ser resolvido hoje. Às vezes, só sentir já é bastante.",
            "en": "Not everything needs to be fixed today. Sometimes, just feeling is enough."
        },
        "acao": {
            "pt": "Respire fundo. Tente escrever três coisas pequenas que te fizeram bem esta semana.",
            "en": "Take a deep breath. Try writing down three small things that made you feel good this week."
        },
        "pergunta": {
            "pt": "O que você diria a uma amiga que estivesse se sentindo assim?",
            "en": "What would you say to a friend feeling this way?"
        }
    },
    "anger": {
        "reflexao": {
            "pt": "A raiva é só uma forma da dor colocar armadura. O que está machucando por baixo dela?",
            "en": "Anger is just pain wearing armor. What’s hurting underneath it?"
        },
        "acao": {
            "pt": "Saia um pouco. Beba água. Bata um travesseiro, se precisar. O corpo também sente.",
            "en": "Step outside. Drink some water. Punch a pillow if needed — your body feels it too."
        },
        "pergunta": {
            "pt": "Sua raiva está te protegendo de algo que você não está querendo sentir?",
            "en": "Is your anger protecting you from something you’re afraid to feel?"
        }
    },
    "joy": {
        "reflexao": {
            "pt": "A felicidade não precisa de explicação. Ela se sustenta no instante.",
            "en": "Happiness doesn’t need explanation. It lives in the moment."
        },
        "acao": {
            "pt": "Guarde esse momento. Tira uma foto mental e agradeça, mesmo que baixinho.",
            "en": "Savor this moment. Take a mental picture and whisper your gratitude."
        },
        "pergunta": {
            "pt": "O que você pode fazer hoje para espalhar essa alegria para alguém?",
            "en": "What can you do today to share this joy with someone?"
        }
    },
    "optimism": {
        "reflexao": {
            "pt": "Esperar o melhor é um ato de coragem disfarçado de fé.",
            "en": "Hoping for the best is an act of courage disguised as faith."
        },
        "acao": {
            "pt": "Escreva um plano simples para algo que você vem adiando — só o primeiro passo.",
            "en": "Write a simple plan for something you’ve been postponing — just the first step."
        },
        "pergunta": {
            "pt": "O que está fazendo você acreditar que as coisas vão melhorar?",
            "en": "What’s making you believe things will get better?"
        }
    }
}

def analisar_emocao(texto):
    inputs = tokenizer(texto, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=1).squeeze().numpy()
    labels = ['anger', 'joy', 'optimism', 'sadness']
    emotion_idx = int(np.argmax(probs))
    return labels[emotion_idx], probs[emotion_idx]

def gerar_resposta(emocao, lang="pt"):
    if emocao not in respostas_emocionais:
        return (
            "Não sei como reagir a esse sentimento ainda... mas estou aqui." if lang == "pt"
            else "I don't know how to respond to that feeling yet... but I'm here."
        ), "", ""
    item = respostas_emocionais[emocao]
    return item["reflexao"][lang], item["acao"][lang], item["pergunta"][lang]

texto = st.text_area("Escreva como você está se sentindo hoje:" if lang == "pt" else "Tell me how you're feeling today:")

palavras_tristeza = [
    "morreu", "morte", "perdi", "luto", "saudade", "sozinha", "sozinho",
    "triste", "chorei", "choro", "doeu", "dor", "vazio", "depressão", "desânimo",
    "abatida", "abatido", "exausta", "exausto", "cansada", "cansado",
    "insuportável", "injusto", "solidão", "falta", "abandono", "desisto",
    "acabei", "acabou", "sumiu", "terminou", "quebrou", "partiu", "tristeza",
    "angústia", "angustiada", "angustiado", "desespero"
]
palavras_risco_pt = [
    "me matar", "tirar minha vida", "acabar com tudo", "sumir pra sempre",
    "me suicidar", "quero morrer", "não quero mais viver", "me jogar",
    "desistir de tudo", "acabar com a dor", "dar fim a isso"
]
palavras_risco_en = [
    "kill myself", "end my life", "end it all", "disappear forever",
    "commit suicide", "i want to die", "don’t want to live anymore", "jump off",
    "give up on everything", "stop the pain", "put an end to it"
]

def detectar_risco(texto):
    texto = texto.lower()
    frases = palavras_risco_pt if lang == "pt" else palavras_risco_en
    for frase in frases:
        if frase in texto:
            return True
    return False

def detectar_tristeza_manual(texto):
    for palavra in palavras_tristeza:
        if palavra in texto.lower():
            return True
    return False

if texto:
    if detectar_risco(texto):
        if lang == "pt":
            st.error("⚠️ Detectamos sinais de que você pode estar passando por um momento muito difícil.")
            st.markdown("Você não está sozinha(o). Conversar com alguém pode ajudar.\n\nSe estiver no Brasil, você pode ligar gratuitamente para o **188** (CVV – 24h).")
        else:
            st.error("⚠️ We detected signs that you might be going through a very difficult time.")
            st.markdown("You're not alone. Talking to someone can help.\n\nIf you're in Brazil, you can call **188** for free (CVV – 24h).")
        st.stop()

    emocao, confianca = analisar_emocao(texto)
    if emocao == "joy" and detectar_tristeza_manual(texto):
        emocao = "sadness"
    titulo_emocao = "🧠 Emoção detectada:" if lang == "pt" else "🧠 Detected emotion:"
    st.markdown(f"**{titulo_emocao}** `{emocao.capitalize()}` ({confianca*100:.2f}%)")

    reflexao, acao, pergunta = gerar_resposta(emocao, lang=lang)

    st.subheader("💭 Reflexão" if lang == "pt" else "💭 Reflection")
    st.write(reflexao)

    st.subheader("🧩 Pequena ação" if lang == "pt" else "🧩 Small action")
    st.write(acao)

    st.subheader("🔎 Pergunta para você pensar" if lang == "pt" else "🔎 A question for reflection")
    st.write(pergunta)
