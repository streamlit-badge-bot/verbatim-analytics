#Packages
import streamlit as st
import pandas as pd
import plotly.express as px
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer, SnowballStemmer
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
import string
import re
import json
from nltk.corpus import wordnet
import base64
import SessionState

@st.cache(suppress_st_warning=True, persist=True, show_spinner=False, allow_output_mutation=True)
def process_data(data, content_col):
    with st.spinner("Please Wait  - Cleaning and Processing Data..."):
        data["CLEANED"] = data[content_col].apply(clean_text)
    return data

#Initiate classes
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

#All stopwords
STOPWORDS = set(stopwords.words('english'))
engtl_stop_words = []
for words in open(r'English-Tagalog Stopwords.txt', 'r'):
    engtl_stop_words.append(words.strip().lower())

with open(r'shortcuts_dict.txt', 'r') as s:
    data = s.read()
shortcut = json.loads(data)


def substitutions(text):
    #remove all symbols except those in retain list
    retain_symbols_list = ['%', "#", "@", "_","/"]
    omit_symbols_list = [punc for punc in string.punctuation if punc not in retain_symbols_list]
    text = "".join(u for u in text if u not in omit_symbols_list)

    text = text.replace("'","")
    #replace numbered percentages with X% to represent rates etc.
    text = re.sub(r"[.a-zA-Z0-9]+%", "x%", text)
    
    # Replace all # with 'number'
    text = re.sub(r"#", " number ", text)
    
    #replace those with dates
    text = re.sub(r"(^\d{1,2}\/\d{1,2}\/\d{4}$)|(((\d{2})|(\d))\/((\d{2})|(\d))\/((\d{4})|(\d{2})))|(^[0,1]?\d{1}\/(([0-2]?\d{1})|([3][0,1]{1}))\/(([1]{1}[9]{1}[9]{1}\d{1})|([2-9]{1}\d{3}))$)", "[date]", text)

    # Replace all / with spaced character to seperate words
    text = re.sub(r"/", " ", text)   
    
    # Replace 89100 with hotline
    text = re.sub(r"(89100)|(8900)", " hotline ", text)   

    #replace those with php,p,k to [amount]
    text = re.sub(r"php([.0-9]+)|php ([.0-9]+)|p([.0-9]+)|p ([.0-9]+)|([.0-9]+)k|([.0-9]+) k|p([.0-9]+)k|php([.0-9]+)k", "[amount]", text)  
    return text

def remove_stopwords(text):
    """Remove stopwords based on nltk english word list and own list"""
    text = ' '.join(word for word in text.split() if word not in STOPWORDS and word not in engtl_stop_words)
    return text

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

@st.cache(suppress_st_warning=True, persist=True, show_spinner=False)
def clean_text(text):
    try:          
        text = substitutions(text.strip().lower())
        text = ' '.join(shortcut[word] if word in shortcut else word for word in text.split()) 
        text = remove_stopwords(text)
        text = ' '.join(lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(text))
        return text
    except:
        return text

def vectorize(data, ngram=(1,3)):
        count_vectorizer = CountVectorizer(stop_words='english', ngram_range=ngram, binary=True, min_df=2, strip_accents='unicode')
        count_data = count_vectorizer.fit_transform(data["CLEANED"].dropna().astype(str))
        return count_vectorizer, count_data
 
def get_topwords(data, count_vectorizer, count_data):
    #Extract and pad features
    vector = pd.DataFrame(zip(count_vectorizer.get_feature_names(), count_data.sum(0).getA1()))
    vector.columns = ['word','count']
    vector = vector.sort_values(by='count', ascending=False)
    return vector

def plot_top_words(plotdf, N=10, color="Reds", subtitle="Overall"):
    condition = plotdf["count"].nlargest(N)
    fig = px.bar(plotdf[plotdf["count"].isin(condition)].sort_values(by='count'),
                title=f"Top {N} Keywords - {subtitle}",
                x='count',
                y='word',
                text='count',
                orientation='h',
                color="count",
                color_continuous_scale=color,
                labels={"word":""}
                )
    fig.for_each_trace(lambda t: t.update(textfont_color="black", textposition='outside'))
    fig.update_layout(coloraxis_showscale=False, template="plotly_white",
                    xaxis={'visible': False, 'showticklabels': False}, title_x=0.5, autosize=True, margin=dict(pad=10))
    return fig

def load_topwords(data, ngram=(1,3)):
    count_vectorizer, count_data = vectorize(data, ngram)
    plotdf = get_topwords(data, count_vectorizer, count_data)
    return plotdf

def word_viewer(data, words, content_col):
	df_viewed = data[data["CLEANED"].str.contains("|".join(w for w in words) , na=False, case=False, regex=False)][[content_col]].reset_index(drop=True)
	st.markdown(get_table_download_link(df_viewed), unsafe_allow_html=True)
	return st.table(df_viewed)

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    # href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    href = f'<a href="data:file/csv;base64,{b64}" download="table.csv">Click here to download table</a>'
    return href

def load_topwords_page(data):
    st.header("📙 Top Keywords")
    st.markdown("* This feature allows you to extract the top keywords in the selected text column. \
                \n* With the ngrams functionality, it can include one to three consecutive words in the sentence for a better overview of content. \
                    \n* The frequency reflected in the tables and charts below represents the number of customers/rows that mentioned the listed word/s in their response.")
    
    #Column Selections
    columns = [col for col in data.columns]
    content_col = st.sidebar.radio("Select Text Column", (columns))
    filter_col = st.sidebar.radio("Select Category Column", (['None'] + columns))
    
    session_state = SessionState.get(checkboxed=False)
    if st.sidebar.button("Confirm") or session_state.checkboxed:
        session_state.checkboxed = True
        pass
    else:
        st.stop()     
    #Preprocessing
    data = process_data(data, content_col)
    
    #Info
    rows = data.shape[0]
    with_content = data[content_col].dropna().shape[0]
    st.subheader(f"Of the {rows:,} total responses received, only {with_content:,} had been analyzed")

    #Plot
    color = st.selectbox("Select Color Palette: ", ("Reds","Oranges", "YlOrBr","YlOrRd","inferno","OrRd","RdGy","Greys"))
    topn = st.slider("Display Top N keywords: ", min_value=10, max_value=20, step=1)
    if filter_col=='None':
        plotdf = load_topwords(data)
        st.write(f"Word Bank - {(plotdf.shape[0]):,} keywords found")
        wordbank = plotdf[plotdf['count']>1].reset_index(drop=True)
        st.dataframe(wordbank)
        st.markdown(get_table_download_link(wordbank), unsafe_allow_html=True)
        select_words = st.multiselect("Select words to remove: ", options=(plotdf["word"].values))
        
        if len(select_words)>0:
            mask = plotdf['word'].isin(select_words)
            new_plot_df = plotdf[~mask]
            st.plotly_chart(plot_top_words(new_plot_df, N=int(topn), color=color, subtitle="Overall"))
        else:
            st.plotly_chart(plot_top_words(plotdf, N=int(topn), color=color, subtitle="Overall"))
        
        with st.beta_expander("- See Top N-grams"):
        #Unigram
            uni_df = load_topwords(data, ngram=(1,1))
            one_word = st.multiselect("Select words to remove: ", options=(uni_df["word"].values))
            st.markdown("- Unigram (one-word)")
            if len(one_word)>0:
                mask = uni_df['word'].isin(one_word)
                new_uni_df = uni_df[~mask]
                st.plotly_chart(plot_top_words(new_uni_df, N=int(topn), color=color, subtitle="Overall Unigram"))
            else:
                st.plotly_chart(plot_top_words(uni_df, N=int(topn), color=color, subtitle="Overall Unigram"))
        #Bigram
            bi_df = load_topwords(data, ngram=(2,2))
            two_word = st.multiselect("Select words to remove: ", options=(bi_df["word"].values))
            st.markdown("- Bigram (two-word)")
            if len(two_word)>0:
                mask = bi_df['word'].isin(two_word)
                new_bi_df = bi_df[~mask]
                st.plotly_chart(plot_top_words(new_bi_df, N=int(topn), color=color, subtitle="Overall Bigram"))
            else:
                st.plotly_chart(plot_top_words(bi_df, N=int(topn), color=color, subtitle="Overall Bigram"))
        #Trigram
            tri_df = load_topwords(data, ngram=(3,3))
            three_word = st.multiselect("Select words to remove: ", options=(tri_df["word"].values))
            st.markdown("- Trigram (three-word)")
            if len(three_word)>0:
                mask = tri_df['word'].isin(three_word)
                new_tri_df = tri_df[~mask]
                st.plotly_chart(plot_top_words(new_tri_df, N=int(topn), color=color, subtitle="Overall Trigram"))
            else:
                st.plotly_chart(plot_top_words(tri_df, N=int(topn), color=color, subtitle="Overall Trigram"))
        
        # Chart Maker
        st.subheader("Word Visualizer")
        retain_words = st.multiselect("Search words to display: ", options=(plotdf["word"].values))
        if len(retain_words)>0:
            mask = plotdf['word'].isin(retain_words)
            new_viz_df = plotdf[mask]
            st.plotly_chart(plot_top_words(new_viz_df, N=int(topn), color=color, subtitle=""))

        # Word Viewer
        st.subheader("Word Viewer")
        words = st.multiselect("Search content with the following words: ", options=(plotdf["word"].values))
        if len(words)>0:
            word_viewer(data, words, content_col)
    else:
        for choices in data[filter_col].unique().tolist():
            st.info(choices)
            cat_df = data[data[filter_col]==choices]
            plotdf = load_topwords(cat_df)
            remove_words = st.multiselect("Select words to remove: ", options=(plotdf["word"].values))
            if len(remove_words)>0:
                mask = plotdf['word'].isin(remove_words)
                new_plotdf = plotdf[~mask]
                st.plotly_chart(plot_top_words(new_plotdf, N=int(topn), color=color, subtitle=choices))
            else:
                st.plotly_chart(plot_top_words(plotdf, N=int(topn), color=color, subtitle=choices))
                
        # Chart Maker
        st.subheader("Word Visualizer")
        retain_words = st.multiselect("Search words to display: ", options=(plotdf["word"].values))
        if len(retain_words)>0:
            mask = plotdf['word'].isin(retain_words)
            new_viz_df = plotdf[mask]
            st.plotly_chart(plot_top_words(new_viz_df, N=int(topn), color=color, subtitle=""))

        # Word Viewer
        st.subheader("Word Viewer")
        plotdf = load_topwords(data)
        words = st.multiselect("Search content with the following words: ", options=(plotdf["word"].values))
        if len(words)>0:
            word_viewer(data, words, content_col)