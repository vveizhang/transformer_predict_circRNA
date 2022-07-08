import streamlit as st
from gpt2_funcs import load_model, load_tokenizer, read_input, run_model

st.title("Transformer circRNA lincRNA Classifier")
st.write('This app uses Transformer to identify the category of any input non-coding RNA sequences')
st.markdown('The source code for this app can be found in this GitHub repo:(https://github.com/vveizhang/transformer_predict_circRNA).')

example_text = """
"GCATGTTGGCATTGAACATTGACGAAGCTATTACATTGCTTGAACAATTGGGACTTAGTGGCAGCTATCAATGGTGTAATACCACAGGATGGCATTCTACAAAGTGAATATGGAGGTGAGACCATACCAGGACCTGCATTTAATCCAGCAAGTCATCCAGCTTCAGCTCCTACTTCCTCTTCTTCTTCAGCGTTTCGACCTGTAATGCCATCCAGGCAGATTGTAGAAAGGCAACCTCGGATGCTGGACTTCAGGGTTGAATACAGAGACAGAAATGTTGATGTGGTACTTGAAGACACCTGTACTGTTG"
"""

input_text = st.text_area(
    label="Input/Paste News here:",
    value="",
    height=30,
    placeholder="Example:{}".format(example_text)
    )

# load model here to save
model = load_model(path="./model/transformer-rna-model.pth")

if input_text == "":
    input_text = example_text

if st.button("Run Transformer!"):
    if len(input_text) < 300:
        st.write("Please input more text!")
    else:
        with st.spinner("Running..."):

            model_input = read_input(tokenizer, input_text)
            model_output = run_model(model, *model_input)
            st.write("Predicted RNA Category (1 as circRNA, 0 as lincRNA):")
            st.write(model_output)
