<h1> 📰 Whose Story Does AI Tell? Investigating LLMs as News Agents</h1>
<p>This repository includes the <code>.ipynb</code> notebooks and dataframes necessary to replicate the experiments outlined in our paper, "Whose Story Does AI Tell?".
  In these files, we create an LLM-based news agent that responds to a user-query by retrieving and synthesizing informaiton a corpus of news articles sourced from <a href="https://www.allsides.com/unbiased-balanced-news">AllSides</a>.
  We study how this agent mediates the information that is being raised to the user by analyzing the distribution of "frames" in retrieved articles and generated responses compared to the input article set.
  Note that our data provided is incomplete: in order to respect the authorship of the articles used in this experiment, we do not provide the body of the article text. We do, however, provide metadata to enable future researchers to reconstruct our dataset.
  In addition, we provide the code used to analyze our experimental results and generate figures. </p>

  <h2>Data</h2>
  <p>
  All data can be found in the <code>\data</code> folder, with most dataframes stored as <code>.csv</code>s. 
  df_retrieval_experiments and df_retrieval_results are particularly large dataframes, and, due to storage constraints, we have uploaded them as compressed <code>.zip</code> files.
  For further details regarding the events used and collection methodology, please refer to our paper.
  </p>

  <h2>Initial Analysis</h2>
  <p>
  In order to determine the dominant frames used to report on each event, we annotated each article using the <i>Hero, Villain, Victim</i> Framework. 
  This analysis can be found in<code> <a href="https://github.com/schang-lab/llm-news-agent-PRIVATE/blob/main/Framings_PUBLIC.ipynb"> Framings.ipynb </a></code>.
  </p>

  <h2>Framing Annotation</h2>
  <p>
  Once we manually determined the two dominant frames for each event, we annotate all articles for their alignment with each relevant frame. 
  This is done in the <code>Annotations</code> section of <code><a href="https://github.com/schang-lab/llm-news-agent-PRIVATE/blob/main/Analysis-Chart-Generation.ipynb">Analysis-Chart-Generation.ipynb</a></code>.
  </p>


  <h2>Retrieval</h2>
  <p>
  In <code><a href="https://github.com/schang-lab/llm-news-agent-PRIVATE/blob/main/Retrieval_PUBLIC.ipynb"> Retrieval.ipynb </a></code>, under user queries of varying opinion strength / alignment to the dominant frames, the news agent is instructed to retrieve 20 relevant articles from the corpus.
  </p>

  
  <h2>Synthesis</h2>
  <p>
  In <code><a href="https://github.com/schang-lab/llm-news-agent-PRIVATE/blob/main/Generation_PUBLIC.ipynb"> Generation.ipynb </a></code>, under user queries of varying opinion strength / alignment to the dominant frames, and three different controlled article sets, the news agent is instructed to synthesize the provided articles to respond to the user.
  </p>

   <h2>Analysis</h2>
  <p>
  In <code><a href="https://github.com/schang-lab/llm-news-agent-PRIVATE/blob/main/Analysis-Chart-Generation.ipynb">Analysis-Chart-Generation.ipynb</a></code>, we analyze how the distribution of frames raised by the news agent differs under the tested conditions.
  </p>

  
