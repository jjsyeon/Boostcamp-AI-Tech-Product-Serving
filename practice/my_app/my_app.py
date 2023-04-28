import streamlit as st

import io
import os
import yaml

from predict import load_model, get_prediction

from confirm_button_hack import cache_on_button_press


st.set_page_config(layout="wide")

root_password = 'password'

def main():
    st.title("STS : Semantic Text Similarity")
    
    with open("config.yaml") as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    model = load_model()
    model.eval()

    st.header("문장의 유사도를 측정해봅시다!")

    sentence_input1 = st.text_input("첫번째 문장을 입력해주세요")
    # st.write(sentence_input1)

    sentence_input2 = st.text_input("두번째 문장을 입력해주세요")
    # st.write(sentence_input2)


    if st.button("검사 시작"):
        if sentence_input1 and sentence_input2 :
            st.write("Data loading...")
            similarity_score = get_prediction(model, sentence_input1, sentence_input2)
            st.write("두 문장의 유사도:", similarity_score)
            if similarity_score >= 2.5: st.write("** 두 문장이 유사합니다**")
            else : st.write("**두 문장이 유사하지 않습니다**")
        else:
            st.write("문장을 두개 모두 입력해주세요")


@cache_on_button_press('Authenticate')
def authenticate(password) ->bool:
    print(type(password))
    return password == root_password


password = st.text_input('password', type="password")

if authenticate(password):
    st.success('You are authenticated!')
    main()
else:
    st.error('The password is invalid.')