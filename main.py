import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="My Deary", page_icon="🐧", layout="centered")

with open("styles/theme.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.image("assets/mascot.png", width=120, caption="Oi! Eu sou o Deary 🐧", use_column_width=False)

st.title("Um amigo para seus desabafos 💗")
st.markdown("_Versão em testes – aqui pra ouvir você com carinho._")

MODEL_NAME = "cardiffnlp/twitter-roberta-base-emotion"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

respostas_emocionais = {
    "sadness": {
        "reflexao": "Tudo que pesa agora não precisa ser resolvido hoje. Às vezes, só sentir já é bastante.",
        "acao": "Respire fundo. Tente escrever três coisas pequenas que te fizeram bem esta semana.",
        "pergunta": "O que você diria a uma amiga que estivesse se sentindo assim?"
    },
    "anger": {
        "reflexao": "A raiva é só uma forma da dor colocar armadura. O que está machucando por baixo dela?",
        "acao": "Saia um pouco. Beba água. Bata um travesseiro, se precisar. O corpo também sente.",
        "pergunta": "Sua raiva está te protegendo de algo que você não está querendo sentir?"
    },
    "joy": {
        "reflexao": "A felicidade não precisa de explicação. Ela se sustenta no instante.",
        "acao": "Guarde esse momento. Tira uma foto mental e agradeça, mesmo que baixinho.",
        "pergunta": "O que você pode fazer hoje para espalhar essa alegria para alguém?"
    },
    "optimism": {
        "reflexao": "Esperar o melhor é um ato de coragem disfarçado de fé.",
        "acao": "Escreva um plano simples para algo que você vem adiando — só o primeiro passo.",
        "pergunta": "O que está fazendo você acreditar que as coisas vão melhorar?"
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

def gerar_resposta(emocao):
    if emocao not in respostas_emocionais:
        return "Não sei como reagir a esse sentimento ainda... mas estou aqui.", "", ""
    item = respostas_emocionais[emocao]
    return item["reflexao"], item["acao"], item["pergunta"]


texto = st.text_area("Escreva como você está se sentindo hoje:")

palavras_tristeza = [
    "morreu", "morte", "perdi", "luto", "saudade", "sozinha", "sozinho",
    "triste", "chorei", "choro", "doeu", "dor", "vazio", "depressão", "desânimo",
    "abatida", "abatido", "exausta", "exausto", "cansada", "cansado",
    "insuportável", "injusto", "solidão", "falta", "abandono", "desisto",
    "acabei", "acabou", "sumiu", "terminou", "quebrou", "partiu", "tristeza",
    "angústia", "angustiada", "angustiado", "desespero"
]
palavras_risco = [
    "me matar", "tirar minha vida", "acabar com tudo", "sumir pra sempre",
    "me suicidar", "quero morrer", "não quero mais viver", "me jogar",
    "desistir de tudo", "acabar com a dor", "dar fim a isso"
]

def detectar_risco(texto):
    texto = texto.lower()
    for frase in palavras_risco:
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
        st.error("⚠️ Detectamos sinais de que você pode estar passando por um momento muito difícil.")
        st.markdown("Você não está sozinha(o). Conversar com alguém pode ajudar.\n\nSe estiver no Brasil, você pode ligar gratuitamente para o **188** (CVV – 24h).")
    st.stop()

    emocao, confianca = analisar_emocao(texto)
    if emocao == "joy" and detectar_tristeza_manual(texto):
        emocao = "sadness"
    st.markdown(f"**🧠 Emoção detectada:** `{emocao.capitalize()}` ({confianca*100:.2f}%)")

    reflexao, acao, pergunta = gerar_resposta(emocao)

    st.subheader("💭 Reflexão")
    st.write(reflexao)

    st.subheader("🧩 Pequena ação")
    st.write(acao)

    st.subheader("🔎 Pergunta para você pensar")
    st.write(pergunta)
