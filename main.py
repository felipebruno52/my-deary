import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Carregar modelo e tokenizer
MODEL_NAME = "cardiffnlp/twitter-roberta-base-emotion"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Dicionário de respostas empáticas
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

# Função de análise
def analisar_emocao(texto):
    inputs = tokenizer(texto, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=1).squeeze().numpy()
    labels = ['anger', 'joy', 'optimism', 'sadness']
    emotion_idx = int(np.argmax(probs))
    return labels[emotion_idx], probs[emotion_idx]

# Função para gerar resposta empática
def gerar_resposta(emocao):
    if emocao not in respostas_emocionais:
        return "Não sei como reagir a esse sentimento ainda... mas estou aqui.", "", ""
    item = respostas_emocionais[emocao]
    return item["reflexao"], item["acao"], item["pergunta"]

# Interface Streamlit
st.set_page_config(page_title="My Deary")
st.title("Um amigo para seus desabafos. 'Uma versão em testes! :)'")

texto = st.text_area("Escreva como você está se sentindo hoje:")

if texto:
    emocao, confianca = analisar_emocao(texto)
    st.markdown(f"**🧠 Emoção detectada:** `{emocao.capitalize()}` ({confianca*100:.2f}%)")

    reflexao, acao, pergunta = gerar_resposta(emocao)

    st.subheader("💭 Reflexão")
    st.write(reflexao)

    st.subheader("🧩 Pequena ação")
    st.write(acao)

    st.subheader("🔎 Pergunta para você pensar")
    st.write(pergunta)
