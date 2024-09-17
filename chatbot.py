import streamlit as st
import ast
import cohere
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from annoy import AnnoyIndex
import pytesseract
from pdf2image import convert_from_bytes
from multiprocessing import Pool, cpu_count
import graphviz
from spire.pdf.common import *
from spire.pdf import *
from PIL import Image
import os
import pytesseract
from io import BytesIO

co = cohere.Client(api_key='002erTCoOQSse0A7PO6nKWGfkHOc8W7G1cr7u4Y2')

# Define process_image function outside of pdf_to_text
def process_image(image):
    return pytesseract.image_to_string(image)

def pdf_to_text(pdf):
    global texts
    # Convert PDF pages to images
    images = convert_from_bytes(pdf.read())
    
    # Perform OCR on each image
    texts = []

    # Create a progress bar
    progress_bar = st.progress(0)
    total_pages = len(images)
    
    # Use multiprocessing for OCR
    with Pool(processes=cpu_count()) as pool:
        for i, text in enumerate(pool.imap(process_image, images)):
            texts.append(text)
            progress = (i + 1) / total_pages
            progress_bar.progress(progress)

    progress_bar.empty()
    
    # Join all texts and split into paragraphs
    full_text = '\n\n'.join(texts)
    
    # Write to file (if needed)
    with open(f'{pdf.name}_output.txt', 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    # Split into paragraphs and clean up
    paragraphs = [t.strip() for t in full_text.split('\n\n') if t.strip()]
    
    st.subheader("Text extracted")
    return np.array(paragraphs)

def search(query, top_k=5):
    global search_index, texts
    # Get the query's embedding
    query_embed = co.embed(texts=[query]).embeddings
    
    # Retrieve the top 'k' nearest neighbors
    similar_item_ids, distances = search_index.get_nns_by_vector(query_embed[0], top_k, include_distances=True)
    
    # Fetch the corresponding text paragraphs
    search_results = [texts[idx] for idx in similar_item_ids]
    
    return search_results

def generate_flowchart(steps):
    dot = graphviz.Digraph()
    dot.attr(rankdir='TB')
    dot.attr('node', fontsize='16')  # Set font size to 16
    
    for i, step in enumerate(steps):
        # Truncate long steps to improve readability
        truncated_step = step[:50] + '...' if len(step) > 50 else step
        dot.node(f'step_{i}', truncated_step)
        if i > 0:
            dot.edge(f'step_{i-1}', f'step_{i}')
    
    return dot

# ## Generating Answers with RAG
def ask_rag(question, num_generations=5):
    global search_index, texts
    
    # Step 1: Retrieval
    retrieved_chunks = search(question, top_k=5)
    
    # Concatenate retrieved chunks to form the context
    context = "\n\n".join(retrieved_chunks)
    
    # Step 2: Generation
    prompt = f"""
    context: 
    {context}
    Question: {question}
    
Provide an answer to the question only when answer is on the given context ,without any grammatical errors. Follow these instructions:

    1. For lists:
       - Present information in a clear, bulleted or numbered list.
       - Use standard markdown list formatting (e.g., "- " for bullets, "1. " for numbers).
       - Do not use any special characters or emojis as list markers.

    2. For tables:
       - Organize information in a structured table using markdown.
       - Use the following format:
         | Column 1 | Column 2 | Column 3 |
         |----------|----------|----------|
         | Data 1   | Data 2   | Data 3   |
       - Ensure proper alignment and consistent use of | and - characters.

    3. For detailed explanations:
       - Provide a comprehensive answer with relevant examples or elaborations from the context.
       - Use clear paragraph breaks for readability.

    4. If the context does not contain the answer:
       - Clearly state that the answer is not available in the given context.

    5. For steps or processes:
       - Number each step clearly using standard markdown numbering (e.g., "1. ", "2. ").

    Tailor your response to the specific request in the question (listing, tabulating, explaining in detail, or describing steps). Ensure all formatting is consistent and clean, without any unnecessary characters or symbols.
if answer is not available return that the answer is not available
    """
    
    prediction = co.generate(
        prompt=prompt,
        max_tokens=300,  # Increased to allow for more detailed responses
        model="command-nightly",
        temperature=0.5,
        num_generations=num_generations
    )

    return prediction.generations
def extract_images_from_pdf(pdf_path, scale_factor=2):
    
    # Create a PdfDocument object and load the PDF file
    doc = PdfDocument()
    doc.LoadFromFile(pdf_path)

    # Create a PdfImageHelper object
    image_helper = PdfImageHelper()
    
    image_count = 1
    ocr_texts = []  # List to store OCR text for each image
    # Extract the PDF name without extension
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    # Create a new directory with the PDF's name
    output_dir = os.path.join(os.path.dirname(pdf_path), pdf_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Iterate over all pages in the PDF
    for page_index in range(doc.Pages.Count):
        # Get the image information for the current page
        image_infos = image_helper.GetImagesInfo(doc.Pages[page_index])
        
        # Extract and save the images
        for image_index in range(len(image_infos)):
            # Get the image stream
            image_stream = image_infos[image_index].Image
            
            # Use a different attribute for naming, e.g., ImageIndex
            temp_file = os.path.join(output_dir, f"temp_image_{page_index + 1}_{image_index + 1}.png")
      
            # Save the image to a temporary file
            image_stream.Save(temp_file)
            
            # Open the saved image using PIL
            image = Image.open(temp_file)

            # Resize the image
            width, height = image.size
            resized_image = image.resize((int(width * scale_factor), int(height * scale_factor)))

            # Define final output file path
            output_file = os.path.join(output_dir, f"image_{page_index + 1}_{image_index + 1}.png")

            # Save the resized image
            

            # Perform OCR on the resized image
            ocr_text = pytesseract.image_to_string(resized_image)
            #print(ocr_text.strip()=='A')
            if ocr_text!= '' and ocr_text.strip()!='A':
                resized_image.save(output_file)
                ocr_texts.append([page_index+1,image_index+1,ocr_text.split('\n')])  # Append OCR text to the list

            # Optionally, remove the temporary file
            os.remove(temp_file)

            image_count += 1

    # Close the PdfDocument object
    doc.Close()
    
    
    img_keys=[]
    for i in range(len(ocr_texts)):
        if ocr_texts[i][2][0]!='' and ocr_texts[i][2][0]!='A':
            t=[x for x in ocr_texts[i][2] if x != '' ]  
            img_keys.append([ocr_texts[i][0],ocr_texts[i][1],t])   
    with open(f"{pdf_name}_img_keys.txt", "w") as file:
        for img_key in img_keys:
            file.write(f"{img_key}\n")
    #print(*img_keys,sep='\n')
    #print(len(img_keys))


def search_images_by_keyword(img_keys, user_query):
    relevant_img_keys = []
    
    for img_key in img_keys:
        for paragraph in img_key[2]:
            if user_query.lower() in paragraph.lower():
                relevant_img_keys.append(img_key)
                break  # Stop checking once a match is found
    return relevant_img_keys


# UI

global search_index, texts
st.title("ChatPDF with RAG")

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'search_index' not in st.session_state:
    st.session_state.search_index = None
if 'texts' not in st.session_state:
    st.session_state.texts = None
if 'current_directory' not in st.session_state:
    st.session_state.current_directory=None

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Side bar for file uploaders and revisit option
with st.sidebar:
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="uploaded_file")
    search_index_file = st.file_uploader("Choose a pre-built search index file", type="ann", key="search_index_file")
    
    # Add revisit option for last 10 questions
    st.subheader("Revisit Previous Questions")
    last_10_questions = [msg["content"] for msg in st.session_state.chat_history[-20:] if msg["role"] == "user"][-10:]
    if last_10_questions:
        selected_question = st.selectbox("Select a question to revisit:", last_10_questions)
        if st.button("Revisit"):
            user_query = selected_question

if uploaded_file is not None and not st.session_state.pdf_processed:
    texts = pdf_to_text(uploaded_file)
    st.write("PDF uploaded and processed successfully.")
    
    response = co.embed(texts=texts.tolist()).embeddings

    # Build a search index
    embeds = np.array(response)
    
    search_index = AnnoyIndex(embeds.shape[1], 'angular')
    print(embeds.shape)  # This will output something like (n_samples, 1024)
    
    # Add all the vectors to the search index
    with st.spinner("Embedding and indexing in progress..."):
        for i in range(len(embeds)):
            search_index.add_item(i, embeds[i])
    search_index.build(10)  # 10 trees
    search_index.save(f'{uploaded_file.name}_search_index.ann')
    print(type(search_index))
    st.success("Embedding and indexing completed.")
    current_directory = os.getcwd()
    current_directory += os.path.join("\\", uploaded_file.name)  # Use os.path.join for better path handling
    st.session_state.current_directory=current_directory
    print(current_directory)
    extract_images_from_pdf(current_directory, 2)
    img_keys=[]
    with open(f'{uploaded_file.name[:-4]}_img_keys.txt', 'r') as f:
        keys = f.read().splitlines()
        for x in keys:
            a= ast.literal_eval(x)
            img_keys.append(a)
    st.session_state.img_keys = img_keys
    # Store processed data in session state
    st.session_state.pdf_processed = True
    st.session_state.search_index = search_index
    st.session_state.texts = texts

    # Remove the file uploader after processing


elif st.session_state.pdf_processed:
    st.success("PDF already processed. Ready for queries.")
else:
    if search_index_file is not None:
        search_index = AnnoyIndex(4096, 'angular')
        search_index.load(search_index_file.name)
        st.session_state.search_index = search_index
        print(search_index==None)
        with open(f'{search_index_file.name[:-17]}_output.txt', 'r', encoding='utf-8') as f:
            texts = f.read()
        paragraphs = [t.strip() for t in texts.split('\n\n') if t.strip()]
        texts = np.array(paragraphs)
        st.session_state.texts = texts
        img_keys=[]
        with open(f'{search_index_file.name[:-21]}_img_keys.txt', 'r') as f:
            keys = f.read().splitlines()
            for x in keys:
                a= ast.literal_eval(x)
                img_keys.append(a)
        uploaded_file=search_index_file.name[:-21]
        st.session_state.img_keys = img_keys
        current_directory = os.getcwd()
        current_directory += os.path.join("\\", uploaded_file) 
        print(current_directory)
        print(st.session_state.current_directory)
        st.session_state.current_directory=current_directory

if 'img_keys' not in st.session_state:
    st.session_state.img_keys = [] 
# Get user input
if 'user_query' not in locals():
    user_query = st.chat_input("Ask a question:", key="user_query")

if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        search_index = st.session_state.search_index
        texts = st.session_state.texts
        
        # Generate a response with RAG
        responses = ask_rag(user_query, num_generations=1)
        response = responses[0].text
        message_placeholder.markdown(response)
        
        # Check if the response contains numbered steps for a flowchart
        steps = [step.strip() for step in response.split('\n') if step.strip().startswith(tuple(str(i)+'.' for i in range(1,10)))]
        if len(steps) > 1:
            flowchart = generate_flowchart(steps)
            st.graphviz_chart(flowchart)
        # Ask if the user needs a simpler explanation or clarification
    need_clarification = st.button("Do you need a simpler explanation or clarification?")
    
    if need_clarification:
        clarification_prompt = f"""
        User query: {user_query}
        Previous response: {response}
        
        The user has requested a simpler explanation. Please provide a simplified version of the previous explanation, using simpler terms and concepts.
        """
        
        clarification_response = co.generate(
            prompt=clarification_prompt,
            max_tokens=200,
            model="command-nightly",
            temperature=0.5,
            num_generations=1
        )
        
    # Check if user needs clarification or simplification
    clarification_prompt = f"""
    User query: {user_query}
    Previous response: {response}
    
    Does the user's query indicate they need clarification, simplification, or a rephrased explanation? If so, provide a simplified version of the previous explanation. If not, respond with 'No clarification needed.'
    """
    
    clarification_response = co.generate(
        prompt=clarification_prompt,
        max_tokens=200,
        model="command-nightly",
        temperature=0.5,
        num_generations=1
    )
    

    
    clarification = clarification_response.generations[0].text
    
    if need_clarification and "No clarification needed" not in clarification:
        with st.chat_message("assistant"):
            st.markdown("I apologize if my previous explanation wasn't clear. Let me try to simplify it:")
            st.markdown(clarification)
        st.session_state.chat_history.append({"role": "assistant", "content": f"Simplified explanation: {clarification}"})
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    important_word = co.generate(
        prompt=f"""sentence:{user_query}
        task: choose the one word  in the sentence which is most related to biology  and return only that one word """,
        max_tokens=150,
        model="command-nightly",
        temperature=0.5,
        num_generations=1
    )
    q=important_word.generations[0].text
    print("+++",q)
    relevant_img_keys = search_images_by_keyword(st.session_state.img_keys, q)
    print(relevant_img_keys)
    
    
    current_directory=st.session_state.current_directory
    print(current_directory)
    if relevant_img_keys != []:
        st.subheader("Relevant Images:")
        for img_key in relevant_img_keys:
            img_path = f"image_{img_key[0]}_{img_key[1]}.png"
            if st.session_state.pdf_processed:
                img_path= current_directory[:-4]+'\\'+img_path
            else:
                img_path= current_directory+'\\'+img_path
            st.image(img_path)
        

# Clear chat history
if st.button("Clear Chat History"):
    st.session_state.chat_history = []

# Reset PDF processing
if st.button("Reset PDF Processing"):
    st.session_state.pdf_processed = False
    st.session_state.search_index = None
    st.session_state.texts=None
