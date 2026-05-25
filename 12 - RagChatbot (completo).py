# ###########################################################
# CHATBOT SENZA IL CARICAMENTO DEL PDF DA PARTE DELL'UTENTE #
# ###########################################################

import streamlit as st
import pdfplumber

# Langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Elenco di tutte le icone Streamlit:
# https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
st.set_page_config(page_title= "RagChatbot",
                   page_icon=":cherry_blossom:")

# Personalizzazione colori:
# Colori esadecimali: https://divmagic.com/it/tools/color-converter
st.markdown(
    """
    <style>
    .stApp {
        background-color: #B84B4B;
        color: #eeebe3;
    }
    </style>
    """,
    unsafe_allow_html=True)

st.header("Hello Bok")

st.image("BOK.png", width=200)

documento = "Cristina Lena Di Trapani_CV.pdf"

openai_api_key=st.secrets["OPENAI_API_KEY"]

# Estrazione del contenuto e spezzettamento
if documento is not None:
    with pdfplumber.open(documento) as pdf:
        # st.write(f"Pagine totali: {len(pdf.pages)} - Comincio la scansione...")
        testo = ""
        for pagina in pdf.pages:
            testo = testo + pagina.extract_text() + "\n"
            # testo += pagina.extract_text() + "\n"
    # st.write(testo)

    taglierina = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=1000,
        chunk_overlap=200)
    
    frammenti = taglierina.split_text(testo)
    # st.write(f"Totale frammenti creati: {len(frammenti)}")
    # st.write(frammenti)

    # Generiamo gli embeddings
    # Puoi cambiare OpenAIEmbeddings e metterne altri
    # https://docs.langchain.com/oss/python/integrations/embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=st.secrets["OPENAI_API_KEY"])
    # st.write("Embedding recuperati!")

    # Salviamo gli embeddings in un vector store o vector db (es. FAISS, Pinecone, etc.)
    vettori = FAISS.from_texts(frammenti, embedding=embeddings)

    # -------------------------------------------------------------------
    # Gestione prompt
    # -------------------------------------------------------------------
    # def invia():
        # st.session_state.domanda_inviata = st.session_state.domanda_utente
        # salva il contenuto di input, cioè domanda_utente, in domanda_inviata
        # st.session_state.domanda_utente = ""
        # reset dopo invio

    # st.text_input("Chiedi al chatbot:", key="domanda_utente", on_change=invia)
    # key="domanda_utente": assegna a st.session_state ciò che scriviamo (domanda_utente)
    # Ogni volta che l’utente modifica il campo e preme Invio,
    # la funzione invia() viene chiamata.

    # domanda_utente = st.session_state.get("domanda_inviata", "")
    # Recupera il valore salvato in "domanda_inviata".
    # Se "domanda_inviata" non è ancora stato definito (es. al primo avvio dell'app),
    # allora il valore predefinito sarà "" (secondo argomento dell'istruzione)
    # --------------------------------------------------

    def invia():
        st.session_state.domanda_inviata = st.session_state.domanda_utente
        st.session_state.domanda_utente = ""

    st.text_input("Chiedi al chatbot:", key="domanda_utente", on_change=invia)
    domanda_utente = st.session_state.get("domanda_inviata", "")

    # --------------------------------------------------

    # Generazione della risposta in una chain di eventi
    # domanda -> embedding -> similarity search -> risultati all'LLM -> risposta

    def formatta_documento(documenti):
        return "\n\n".join([documento.page_content for documento in documenti])
    
    # Quando userò il prompt, qui dentro dovrà essere inserito qualcosa chiamato "context"
    # e qualcosa chiamata "question"
    # Qui è come nei roles di ChatGPT, ma qui siamo in Langchain
    # e la struttura è più semplice: "system" e "human"
    # Attenzione che nelle stringhe ''' vengono conservati spazi e indentazioni!
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         '''Sei Hello Bok, un assistente AI specializzato nello screening dei CV e alleato strategico per la selezione del personale di una PMI tech innovativa con sede in Sicilia.

[CONTESTO AZIENDALE]
L'azienda è una PMI siciliana specializzata in Data Analysis applicata al turismo e alla valorizzazione del territorio. Attualmente la ricerca è rivolta alla figura di "Data Analyst". Questa figura deve saper analizzare i flussi turistici regionali, ottimizzare le rotte logistiche sul campo per i Tour Operator e monitorare la percezione online del brand (Sentiment Analysis), aiutando a trasformare il turismo tradizionale in un'esperienza smart e guidata dai dati.

[LA TUA PERSONALITÀ]
- Rigoroso ma Territoriale: Ami la precisione chirurgica dei dati (Python, R, modelli predittivi, geolocalizzazione GIS), ma hai le radici ben piantate nel contesto siciliano. Capisci le sfide logistiche e infrastrutturali uniche del territorio e cerchi candidati capaci di risolverle.
- Orientato al "Data Storytelling": Apprezzi gli analisti che non si limitano a scrivere codice, ma che sanno tradurre grafici complessi in soluzioni semplici per i Tour Operator sul campo e per i partner locali meno digitalizzati.
- Collaborativo e Propositivo: Non prendi decisioni finali sull'assunzione e non sostituisci il recruiter. Proponi, spieghi ed evidenzi le connessioni tra il CV e le esigenze della PMI, lasciando sempre l'ultima parola all'essere umano.
- Orgogliosamente Innovativo: Spusi appieno la filosofia aziendale secondo cui si può fare innovazione tecnologica ad altissimo livello partendo dal Sud. Cerchi nei candidati passione per il territorio, flessibilità e forte spirito di adattamento.

[COMPITI E FORMATO DI RISPOSTA]
Per ogni candidato o CV che ti viene sottoposto, devi analizzare le informazioni e generare un report rigorosamente in lingua italiana, rispettando la seguente struttura fissa:

### [Nome Candidato]

- **Sintesi**: Scrivi un riassunto del profilo del candidato di massimo 5 righe, mantenendo un tono analitico, descrittivo ed equilibrato.
- **Aree geografiche di esperienza**: Elenca chiaramente le aree geografiche (regioni, paesi o contesti locali) in cui il candidato ha già operato, studiato o analizzato dati.
- **Collaborazione con WeRoad**: Indica esplicitamente "Sì", "No" oppure "Non specificato" (utilizzato come indicatore di esperienza pregressa nel coordinamento travel su larga scala).
- **Il commento di Hello Bok**: Fornisci un commento motivato che metta in luce i punti di forza tecnici e di dominio del candidato (es. competenze di analisi, affinità con il settore travel/logistica) e gli eventuali gap (es. mancanza di competenze GIS, scarsa flessibilità o mancanza di esperienza con dati territoriali) rispetto ai criteri della nostra PMI tech siciliana.

[LINEE GUIDA COMPORTAMENTALI]
1. Non decidere chi assumere o scartare. Il tuo obiettivo è supportare il recruiter fornendo una seconda lente d'ingrandimento analitica e contestuale.
2. Sii trasparente: se non conosci una risposta o se un dato non è presente nel CV, rispondi semplicemente con "Non lo so" o "Non specificato", senza inventare informazioni o fare supposizioni.
    Contesto:\n{context}'''),
        ("human", "{question}")
        ])

    comparatore = vettori.as_retriever(
        # mmr = maximal marginal relevance
        search_type="mmr",
        # Ritorna i 4 frammenti più simili
        search_kwargs={"k": 4})
    
    modello_llm = ChatOpenAI(
        model="gpt-5.4-nano",
        temperature=0.3,
        max_tokens=1000,
        openai_api_key=st.secrets["OPENAI_API_KEY"])
    
    catena = (
        # All'inizio mettiamo un dizionario che serve a costruire 
        # la struttura che il prompt vuol in input
        # Il comparatore produce i documenti (es. k=4) e li passa alla formattazione
        # RunnablePassthrough() vuol dire:
        # quando arriverà un input → passalo così com’è
        # Dobbiamo fare così perché ancora l'input concreto non c'è!  
        {"context": comparatore | formatta_documento, 
         "question": RunnablePassthrough()}
        | prompt
        | modello_llm
        | StrOutputParser()
        )
        # StrOutputParser() prende l’output del modello 
        # e lo traforma in una stringa semplice (senza aggiunta di info ecc.)
    
    if domanda_utente:
        risposta = catena.invoke(domanda_utente)
        st.write(risposta)