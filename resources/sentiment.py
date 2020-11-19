import pandas as pd
import streamlit as st
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import altair as alt
import base64
import sent_clean
import spacy
from spacy.matcher import PhraseMatcher
import SessionState
import topwords_sent

@st.cache(suppress_st_warning=True, persist=True, show_spinner=False, allow_output_mutation=True)
def read_file(filename_with_ext: str, sheet_name=0, *kwargs):
    ext = filename_with_ext.partition('.')[-1]
    if ext == 'xlsx':
        file = pd.read_excel(filename_with_ext, sheet_name=sheet_name, *kwargs)
        return file
    elif ext == 'csv':
        file = pd.read_csv(filename_with_ext, *kwargs)
        return file
    elif ext == 'sas7bdat':
        file = pd.read_sas(filename_with_ext, *kwargs)
        return file
    else:
        print('Cannot read file: Convert to .xlsx, .csv or .sas7bdat')

@st.cache(suppress_st_warning=True, persist=True, show_spinner=False, allow_output_mutation=True)
def preprocess(df, content_col, cust_col):
    df = sent_clean.init_clean(df, col=content_col)
    #Converts to Sentence and fill with 'blank' those real blank cells
    df['sentence'] = df['init_cleaned'].map({'..':'blank', '...':'blank'}).fillna(df[content_col]).str.split('.')
  	#replace comments with elipsis or .. with BLANK
    df['sentence'] = df['sentence'].fillna('blank')
    new_df = df.explode('sentence').dropna(subset=['sentence']).query("sentence!=''").reset_index()
	
	#Shows tally
    st.markdown(f"No. of Rows: {new_df['index'].nunique():,}")
    st.markdown(f"No. of Customers: {new_df[cust_col].nunique():,}")
    st.markdown(f"Rows w/o comments: {(new_df['sentence']=='blank').sum():,}")
    sent_df = new_df[['sentence', 'index']]

	#Cleans text
    cleaned_df = sent_clean.final_clean(sent_df, col='sentence')
    return cleaned_df, new_df

def run_model(cleaned_df):
	model = pickle.load(open(r'.\resources\model12.pkl','rb'))
	filtered = cleaned_df[['cleaned']].query("(cleaned!='')&(cleaned!='blank')")['cleaned']
	df = cleaned_df[['sentence', 'cleaned','index']]\
    	.merge(pd.DataFrame(model.predict(filtered),
    	 index=filtered.index)\
    	.rename(columns={0:'pred'}),
    			right_index=True,
    			left_index=True,
    			how='left')
	return df

def sent_tally(out, cust_col):
	#total sentiment count
	src = pd.DataFrame(out['pred'].value_counts()).reset_index().rename(columns={'pred':'count', 'index':'sentiment'})
	row_sent = alt.Chart(src).mark_bar().encode(x=alt.X('sentiment', type='nominal', sort=None, axis=alt.Axis(labelAngle=360, title="")),
                                            	y=alt.Y('count', type='quantitative', axis=alt.Axis(title="Count")),color=alt.Color('sentiment:N',scale=alt.Scale(domain=['Negative','Neutral', 'Positive'],range=['firebrick','gray', 'darkgreen'])),tooltip=['sentiment:N', 'count:Q'])
	#venn diagram of sentiment - customer level
	venn = pd.DataFrame(out.fillna('None').groupby([cust_col,'pred'])['pred'].nunique()).unstack('pred').reset_index().fillna(0)
	venn.columns = [venn.columns.values[0][0],venn.columns.values[1][1],venn.columns.values[2][1],venn.columns.values[3][1],venn.columns.values[4][1]]
	venn['sum'] = venn[['Negative', 'Neutral', 'Positive']].sum(axis=1)
	venn['sentiment'] = np.where((venn['sum']==1)&(venn['Negative']==1),'Negative', 'None')
	venn['sentiment'] = np.where((venn['sum']==1)&(venn['Positive']==1),'Positive', venn['sentiment'])
	venn['sentiment'] = np.where((venn['sum']==1)&(venn['Neutral']==1),'Neutral', venn['sentiment'])
	venn['sentiment'] = np.where((venn['sum']>1), 'Mixed', venn['sentiment'])
	venn1 = pd.DataFrame(venn['sentiment'].value_counts()).reset_index().rename(columns={'sentiment':'count', 'index':'sentiment'}).set_index('sentiment').reindex(['Negative','Mixed','Positive','Neutral','None']).reset_index().fillna(0)
	venn1['Percent of Total'] = venn1['count'].divide(venn1['count'].sum())
	cust_sent = alt.Chart(venn1).mark_bar().encode(x=alt.X('sentiment', type='nominal', sort=None, axis=alt.Axis(labelAngle=360, title="")),
                                                y=alt.Y('Percent of Total', type='quantitative', axis=alt.Axis(format='%')),tooltip=['sentiment:N', 'Percent of Total:Q', 'count:Q'])
	fig = cust_sent.properties(height=300, width=300, title='Sentiment by Customer') | row_sent.properties(height=300, width=300, title='All Sentiment')
	st.markdown(get_table_download_link(venn1, "Sentiment by Customer table"), unsafe_allow_html=True)
	st.markdown(get_table_download_link(src, "All Sentiment table"), unsafe_allow_html=True)
	net_sent_score = (int(src.query("sentiment=='Positive'")['count']) - int(src.query("sentiment=='Negative'")['count'])) / int(src.query("sentiment!='Neutral'")['count'].sum())
	st.markdown(f"### Net Sentiment Score: {(net_sent_score*100):,.2f}%")
	return st.altair_chart(fig)

def relatedwords(out):
	out['category_id'] = out['pred'].fillna('Blank').factorize()[0]
	category_id_df = out[['pred', 'category_id']].fillna('Blank').drop_duplicates().sort_values('category_id')
	category_to_id = dict(category_id_df.values)

	tfidf = TfidfVectorizer(sublinear_tf=True,
	                        min_df=5,
	                        norm='l2',
	                        ngram_range=(1, 3),
	                        stop_words='english')
	features = tfidf.fit_transform(out.cleaned).toarray()
	labels = out.category_id
	corr_dict1 = {}

	for Sentiment, category_id in sorted(category_to_id.items()):
	    if Sentiment != 'Blank':
	        features_chi2 = chi2(features, labels == category_id)
	        indices = np.argsort(features_chi2[0])[::-1]
	        feature_names = np.array(tfidf.get_feature_names())[indices]
	        #N-grams
	        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
	        uni_score = [score for v,score in zip(feature_names, features_chi2[0][indices]) if len(v.split(' ')) == 1]
	        uni_pval = [pval for v,pval in zip(feature_names, features_chi2[1][indices]) if len(v.split(' ')) == 1]
	        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
	        bi_score = [score for v,score in zip(feature_names, features_chi2[0][indices]) if len(v.split(' ')) == 2]
	        bi_pval = [pval for v,pval in zip(feature_names, features_chi2[1][indices]) if len(v.split(' ')) == 2]
	        trigrams = [v for v in feature_names if len(v.split(' ')) == 3]
	        tri_score = [score for v,score in zip(feature_names, features_chi2[0][indices]) if len(v.split(' ')) == 3]
	        tri_pval = [pval for v,pval in zip(feature_names, features_chi2[1][indices]) if len(v.split(' ')) == 3]
	        corr_dict1.update({f"{Sentiment}_uni":unigrams,f"{Sentiment}_uni_score":uni_score,f"{Sentiment}_uni_pval":uni_pval,
	                          f"{Sentiment}_bi":bigrams,f"{Sentiment}_bi_score":bi_score,f"{Sentiment}_bi_pval":bi_pval,
	                          f"{Sentiment}_tri":trigrams, f"{Sentiment}_tri_score":tri_score,f"{Sentiment}_tri_pval":tri_pval})
	
	#Count of Words
	count_vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,3))
	count_data = count_vectorizer.fit_transform(out.cleaned)
	vector = pd.DataFrame(zip(count_vectorizer.get_feature_names(), count_data.sum(0).getA1()))
	vector.columns = ['word','count']

	#Plot
	ngrams = ['uni', 'bi', 'tri']
	sents = ['Positive', 'Negative', 'Neutral']
	for sent in sents:
		li = []
		for ngram,name in zip(ngrams, ['Unigram', 'Bigram', 'Trigram']):
			corr = pd.DataFrame({k:v for k,v in corr_dict1.items() if f'{sent}_{ngram}' in k})
			df_ngram = corr[corr[f"{sent}_{ngram}_pval"]<0.05].head(30).merge(vector, left_on=f'{sent}_{ngram}' , right_on='word', how='left')
			bars = alt.Chart(df_ngram).mark_bar().encode(x="count:Q",y=alt.Y('word', type='nominal', sort=None, axis=alt.Axis(title="")),
														 tooltip=['word:N', 'count:Q'],
														  color=alt.condition(alt.datum[f'{sent}_{ngram}_pval']<0.001, alt.value('firebrick'), alt.value('grey')))
			text = bars.mark_text(align='left', baseline='middle', dx=3).encode(text='count:Q')
			fig = (bars + text).properties(height=500, width=200, title=f"{sent} {name}")
			li.append(fig)
		figs = li[0]|li[1]|li[2]
		st.altair_chart(figs)

def relatedwords1(out):
	out['category_id'] = out['pred'].fillna('Blank').factorize()[0]
	category_id_df = out[['pred', 'category_id']].fillna('Blank').drop_duplicates().sort_values('category_id')
	category_to_id = dict(category_id_df.values)

	tfidf = TfidfVectorizer(sublinear_tf=True,
	                        min_df=5,
	                        norm='l2',
	                        ngram_range=(1, 3),
	                        stop_words='english')
	features = tfidf.fit_transform(out.cleaned).toarray()
	labels = out.category_id
	corr_dict = {}
	ngram_list = [i for i in tfidf.get_feature_names()]
	for Sentiment, category_id in sorted(category_to_id.items()):
	    if Sentiment != 'Blank':
	        features_chi2 = chi2(features, labels == category_id)
	        indices = np.argsort(features_chi2[0])[::-1]
	        feature_names = np.array(tfidf.get_feature_names())[indices]
	        #N-grams
	        unigrams = [v for v in feature_names]
	        uni_score = [score for v,score in zip(feature_names, features_chi2[0][indices])]
	        uni_pval = [pval for v,pval in zip(feature_names, features_chi2[1][indices])]
	        corr_dict.update({f"{Sentiment}_sent":unigrams,f"{Sentiment}_score":uni_score,f"{Sentiment}_pval":uni_pval})
            
	#Count of Words
	count_vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,3))
	count_data = count_vectorizer.fit_transform(out.cleaned)
	vector = pd.DataFrame(zip(count_vectorizer.get_feature_names(), count_data.sum(0).getA1()))
	vector.columns = ['word','count']
	#Plot
	sents = ['Positive', 'Negative', 'Neutral']
	li = []
	for sent in sents:
		corr = pd.DataFrame({k:v for k,v in corr_dict.items() if sent in k})
		df_ngram = corr[corr[f"{sent}_pval"]<0.05].head(30).merge(vector, left_on=f'{sent}_sent' , right_on='word', how='left')
		bars = alt.Chart(df_ngram).mark_bar().encode(x="count:Q",y=alt.Y('word', type='nominal', sort=None, axis=alt.Axis(title="")),
													 tooltip=['word:N', 'count:Q'],
													  color=alt.condition(alt.datum[f'{sent}_pval']<0.001, alt.value('firebrick'), alt.value('grey')))
		text = bars.mark_text(align='left', baseline='middle', dx=3).encode(text='count:Q')
		fig = (bars + text).properties(height=500, width=200, title=f"{sent}")
		li.append(fig)
	figs = li[0]|li[1]|li[2]
	st.altair_chart(figs)
	return ngram_list

def get_table_download_link(df, name):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    # href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    href = f'<a href="data:file/csv;base64,{b64}" download="table.csv" style="font-size: 10px">Download {name}</a>'
    return href

def word_viewer(data, words, content_col, cust_col):
    total_pop = data[cust_col].nunique()
    df_viewed = data[data["sentence"].str.contains(words, na=False, case=False, regex=False)][[cust_col ,content_col, 'pred']].drop_duplicates().reset_index(drop=True)
    curr_pop = df_viewed[cust_col].nunique()
    st.markdown(get_table_download_link(df_viewed, "search results"), unsafe_allow_html=True)
    st.markdown(f"There are {curr_pop:,}/{total_pop:,} respondents found.")
    return st.table(df_viewed)

@st.cache(suppress_st_warning=True, persist=True, show_spinner=False, allow_output_mutation=True)
def get_channels(df, channels):
	nlp = spacy.load("en_core_web_sm")
    #Patter Matcher
	def on_match(matcher, doc, id, matches):
		return [nlp.vocab.strings[match_id] for match_id,start, end in matches]
	matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
	for k,v in channels.items():
	    for item in [word.strip() for sentence in v for word in sentence.split(',')]:
	        matcher.add(str(k), on_match, nlp(str(item)))
	
	df['channel'] = df['sentence'].str.lower().map(lambda text: [nlp.vocab.strings[match_id[0]] for match_id in matcher(nlp(text))])
	for i in [j for j in channels.keys()]:
		df[i] = df['channel'].map(lambda x: 1 if str(i) in x else 0)
	return df

def plot_channels(df,channels,cust_col):
	channel_list = [j for j in channels.keys()]
	a = df.groupby([cust_col, 'pred'])[channel_list].sum().clip(0,1).groupby('pred')[channel_list].sum().reindex(['Neutral', 'Positive', 'Negative'])
	b = a.unstack().reset_index().rename(columns={'level_0':'Channel', 0:'count'})
	c = (a.divide(a.sum())*100).unstack().reset_index().rename(columns={'level_0':'Channel', 0:'percent'})
	source = b.merge(c, on=['Channel', 'pred'])
	st.markdown(get_table_download_link(source, "channels table"), unsafe_allow_html=True)
	bars = alt.Chart(source).mark_bar().encode(
	    x=alt.X('count:Q', stack='zero', axis=alt.Axis(title="Count")),
	    y=alt.Y('Channel:N', sort='-x', axis=alt.Axis(title="")),
	    tooltip=['count:Q', 'Channel:N', 'pred:N', 'percent:Q'],
	    color=alt.Color('pred',scale=alt.Scale(domain=['Negative','Neutral', 'Positive'],range=['firebrick','gray', 'darkgreen']), legend=alt.Legend(title="Sentiment"))
	)
    
	text = alt.Chart(source).mark_text(dx=15, dy=0, color='black').encode(
	    x=alt.X('sum(count):Q', stack='zero', axis=alt.Axis(title="Count")),
	    y=alt.Y('Channel:N', sort='-x', axis=alt.Axis(title="")),
	    detail='Channel:N',
	    text=alt.Text('sum(count):Q', format=',.0f')
	)
	fig = (bars + text).properties(title='By Channel')
	return st.altair_chart(fig)

@st.cache(suppress_st_warning=True, persist=True, show_spinner=False, allow_output_mutation=True)
def get_topics(df, topics):
	nlp = spacy.load("en_core_web_sm")
    #Patter Matcher
	def on_match(matcher, doc, id, matches):
		return [nlp.vocab.strings[match_id] for match_id,start, end in matches]
	matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
	for k,v in topics.items():
	    for item in [word.strip() for sentence in v for word in sentence.split(',')]:
	        matcher.add(str(k), on_match, nlp(str(item)))
	df['topics'] = df['sentence'].str.lower().map(lambda text: [nlp.vocab.strings[match_id[0]] for match_id in matcher(nlp(text))]).apply(lambda y: [str('Others')] if len(y)==2 else y)
	df['topics'] = np.where(df['sentence']=='blank', [str('No Feedback')], df['topics'])
	df['pred'] = np.where(df['topics']=='No Feedback', 'Neutral', df['pred'])
	for i in [j for j in topics.keys()] + ['Others', 'No Feedback']:
		df[i] = df['topics'].map(lambda x: 1 if str(i) in x else 0)
	return df


def plot_topics(df, topics, cust_col):
    topic_list = [j for j in topics.keys()] + ['Others', 'No Feedback']
    a = df.groupby([cust_col, 'pred'])[topic_list].sum().clip(0,1).groupby('pred')[topic_list].sum().reindex(['Neutral', 'Positive', 'Negative'])
    b = a.unstack().reset_index().rename(columns={'level_0':'Topic', 0:'count'})
    c = (a.divide(a.sum())*100).unstack().reset_index().rename(columns={'level_0':'Topic', 0:'percent'})
    source = b.merge(c, on=['Topic', 'pred']).query("count>1")
    st.markdown(get_table_download_link(source, "topics table"), unsafe_allow_html=True)
    bars = alt.Chart(source).mark_bar().encode(
	    x=alt.X('count:Q', stack='zero', axis=alt.Axis(title="Count")),
	    y=alt.Y('Topic:N', sort='-x', axis=alt.Axis(title="")),
	    tooltip=['count:Q', 'Topic:N', 'pred:N', 'percent:Q'],
	    color=alt.Color('pred',scale=alt.Scale(domain=['Negative','Neutral', 'Positive'],range=['firebrick','gray', 'darkgreen']),legend=alt.Legend(title="Sentiment"))
	)
    text = alt.Chart(source).mark_text(dx=15, dy=0, color='black').encode(
	    x=alt.X('sum(count):Q', stack='zero', axis=alt.Axis(title="Count")),
	    y=alt.Y('Topic:N', sort='-x', axis=alt.Axis(title="")),
	    detail='Topic:N',
	    text=alt.Text('sum(count):Q', format=',.0f')
	)
    fig = (bars + text).properties(title='By Topic', height=300, width=600)
    return st.altair_chart(fig)

@st.cache(suppress_st_warning=True, persist=True, show_spinner=False, allow_output_mutation=True)
def get_products(df, products):
	nlp = spacy.load("en_core_web_sm")
    #Patter Matcher
	def on_match(matcher, doc, id, matches):
		return [nlp.vocab.strings[match_id] for match_id,start, end in matches]
	matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
	for k,v in products.items():
	    for item in [word.strip() for sentence in v for word in sentence.split(',')]:
	        matcher.add(str(k), on_match, nlp(str(item)))
	
	df['products'] = df['sentence'].str.lower().map(lambda text: [nlp.vocab.strings[match_id[0]] for match_id in matcher(nlp(text))])
	for i in [j for j in products.keys()]:
		df[i] = df['products'].map(lambda x: 1 if str(i) in x else 0)
	return df

def plot_products(df, products, cust_col):
    prod_list = [j for j in products.keys()]
    a = df.groupby([cust_col, 'pred'])[prod_list].sum().clip(0,1).groupby('pred')[prod_list].sum().reindex(['Neutral', 'Positive', 'Negative'])
    b = a.unstack().reset_index().rename(columns={'level_0':'Product', 0:'count'})
    c = (a.divide(a.sum())*100).unstack().reset_index().rename(columns={'level_0':'Product', 0:'percent'})
    source = b.merge(c, on=['Product', 'pred'])
    st.markdown(get_table_download_link(source, "products table"), unsafe_allow_html=True)
    bars = alt.Chart(source).mark_bar().encode(
	    x=alt.X('count:Q', stack='zero', axis=alt.Axis(title="Count")),
	    y=alt.Y('Product:N', sort='-x', axis=alt.Axis(title="")),
	    tooltip=['count:Q', 'Product:N', 'pred:N', 'percent:Q'],
	    color=alt.Color('pred',scale=alt.Scale(domain=['Negative','Neutral', 'Positive'],range=['firebrick','gray', 'darkgreen']), legend=alt.Legend(title="Sentiment"))
	)
    
    text = alt.Chart(source).mark_text(dx=15, dy=0, color='black').encode(
	    x=alt.X('sum(count):Q', stack='zero', axis=alt.Axis(title="Count")),
	    y=alt.Y('Product:N', sort='-x', axis=alt.Axis(title="")),
	    detail='Product:N',
	    text=alt.Text('sum(count):Q', format=',.0f')
	)
    fig = (bars + text).properties(title='By Product')
    return st.altair_chart(fig)
  
def read_keywords():
    keywords = pd.read_excel(r'.\resources\Keywords.xlsx')
    channels = keywords.query("channel_tag==1")[['Topic','Keywords']].set_index('Topic').T.to_dict('list')
    topics = keywords.query("topic_tag==1")[['Topic','Keywords']].set_index('Topic').T.to_dict('list')
    products = keywords.query("product_tag==1")[['Topic','Keywords']].set_index('Topic').T.to_dict('list')
    return channels, topics, products

def get_topwords(data, content_col, columns):
    for items in columns.keys():
        try:
            cond1 = data[items]==1
            cond2 = ~data[content_col].str.lower().str.contains('|'.join([i for i in [items.lower()[:-1],items.lower()+'ing',items.lower()+'s',items.lower()]]))
            topw = topwords_sent.main(data[cond1&cond2], content_col)
            st.markdown(items)
            st.dataframe(topw)
        except:
            pass
        

def load_sentiment_page(df):
    st.header("🌡️ Sentiment")
    st.markdown("* This feature allows you to extract the sentiment of customers in the selected text column.")
    columns = [col for col in df.columns]
    
    content_col = st.sidebar.radio("Select Text Column", (columns))
    cust_col = st.sidebar.radio("Select Customer ID Column", (columns), index=1)
    segment_col = st.sidebar.radio("Select Segment/category Column", (columns), index=2)
    segment_val = st.selectbox("Which segment/category would you like to view?", tuple(['View All'] + df[segment_col].dropna().unique().tolist()))
    score_col = st.sidebar.radio("Select Score Column", (columns), index=3)

    session_state = SessionState.get(checkboxed=False)
    if st.sidebar.button("Confirm") or session_state.checkboxed:
        session_state.checkboxed = True
        pass
    else:
        st.stop() 
    
    # st.markdown("Preprocessing DataFrame")
    cleaned_df, new_df = preprocess(df, content_col, cust_col)
    pred_df = run_model(cleaned_df)
    out = pred_df.drop(columns=['sentence']).merge(new_df, how='right', on='index', copy=False).drop_duplicates(subset=[cust_col,'sentence'])

    #Run Dashboard =====
    if segment_val!='View All':
        segdf = out[out[segment_col]==segment_val]
    else:
        segdf = out.copy()  

    st.info(f"There are {segdf[cust_col].nunique():,} respondents and {segdf['index'].nunique():,} rows")
    #Sentiment
    sent_tally(segdf, cust_col)
    channels, topics, products = read_keywords()
    ch_df = get_channels(segdf, channels)
    prod_df = get_products(ch_df, products)
    top_df = get_topics(prod_df, topics)
    col1, col2 = st.beta_columns(2)
    #Channels
    with col1:
        plot_channels(ch_df, channels, cust_col)
        with st.beta_expander("- Browse topwords"):
            get_topwords(ch_df, content_col, channels)
    #Products
    with col2:
       plot_products(prod_df, products, cust_col)
       with st.beta_expander("- Browse topwords"):
           get_topwords(prod_df, content_col, products)

    #Topics
    plot_topics(top_df, topics, cust_col)
    with st.beta_expander("- Browse topwords"):
        get_topwords(top_df, content_col, topics)

    #Browse
    with st.beta_expander("- Browse responses"):
        col1, col2 = st.beta_columns(2)
        ref = {"Channel":channels, "Product":products, "Topics":topics}
        with col1:
            cat = st.selectbox("Category:", tuple(ref.keys()))
        with col2:
            cat_val = st.selectbox("Subcategory:", tuple([k.strip() for v in ref[cat] for k in v.split(',')]))
        st.table(top_df[top_df[cat_val]==1][[cust_col, content_col, 'pred', cat_val]])
    
    #Related Words
    st.subheader("Related Words to Sentiment")
    st.markdown("This section shows the words related to the sentiments (in order) and their corresponding number of occurences")
    ngram_list = relatedwords1(segdf)
    with st.beta_expander("- See Ngram breakdown"):
    	relatedwords(segdf)
    
    #Download entire table
    st.markdown(get_table_download_link(top_df, "results"), unsafe_allow_html=True)
    
    #Word Viewer
    st.subheader("Word Viewer")
    words = st.text_input("Search content with the following words: ")
    if len(words)>0:
        word_viewer(top_df, str(words).lower(), content_col, cust_col)