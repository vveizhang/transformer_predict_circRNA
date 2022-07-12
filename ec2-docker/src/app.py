import streamlit as st
from funcs import load_model, read_input, run_model

st.title("Transformer non-coding RNA Classifier")
st.write('This app uses Transformer model to identify the category of any input non-coding RNA sequence')
st.markdown('The source code for this app can be found in this GitHub repo: [github.com/vveizhang/transformer_predict_circRNA](https://github.com/vveizhang/transformer_predict_circRNA)')

example_text = """
{"text":
"GCATGTTGGCATTGAACATTGACGAAGCTATTACATTGCTTGAACAATTGGGACTTAGTGGCAGCTATCAATGGTGTAATACCACAGGATGGCATTCTACAAAGTGAATATGGAGGTGAGACCATACCAGGACCTGCATTTAATCCAGCAAGTCATCCAGCTTCAGCTCCTACTTCCTCTTCTTCTTCAGCGTTTCGACCTGTAATGCCATCCAGGCAGATTGTAGAAAGGCAACCTCGGATGCTGGACTTCAGGGTTGAATACAGAGACAGAAATGTTGATGTGGTACTTGAAGACACCTGTACTGTTG"}
"""


input_text = st.text_area(
    label="Input/Paste News here:",
    value="",
    height=30,
    placeholder="Example:{}".format(example_text)
    )

# load model here to save
model = load_model(path="./model/transformer-model.pth")

if input_text == "":
    input_text = example_text

if st.button("Run Transformer!"):
    if len(input_text) < 300:
        st.write("Please input more text!")
    else:
        with st.spinner("Running..."):

            model_input = read_input(input_text)
            model_output = run_model(model_input, model)
            predicted_label = 'circRNA' if model_output == 1.0 else 'lincRNA'
            st.write(model_output)
            st.write(predicted_label)
