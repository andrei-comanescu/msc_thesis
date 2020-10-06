# Master thesis 
The master thesis pertains to the analysis over-time of political polarisation over time on Twitter in regards to controversial topics. The source code in this repository represents a somewhat curated version of the code used in the thesis.   

The political polarisation of users is estimated using the publically available [MCMC Bayesian Point Estimations](https://github.com/pablobarbera/twitter_ideology) and [GraphSAGE](http://snap.stanford.edu/graphsage/), an inductive representation learning framework for graphs (via the keras library [StellarGraph](https://github.com/stellargraph/stellargraph)). Access to the [Twitter API](https://developer.twitter.com/en/docs) is required to link the user screen names to their IDs (used in the point estimations).


### Thesis abstract:
Social networks represent a public forum of discussion for various topics, some of them controversial.
Twitter is such a social network; it acts as a public space where discourse occurs. In recent years
the role of social networks in information spreading has increased. As have the fears regarding
the increasingly polarised discourse on social networks, caused by the tendency of users to avoid
exposure to opposing opinions, while increasingly interacting with only like-minded individuals.
This work looks at controversial topics on Twitter, over a long period of time, through the prism
of political polarisation. We use the daily interactions, and the underlying structure of the whole
conversation, to create daily graphs that are then used to obtain daily graph embeddings. We
estimate the political ideologies of the users that are represented in the graph embeddings. By
using the political ideologies of users and the daily graph embeddings, we offer a series of methods
that allow us to detect and analyse changes in the political polarisation of the conversation. This
enables us to conclude that, during our analysed time period, the overall polarisation levels for
our examined controversial topics have stagnated. We also explore the effects of topic-related
controversial events on the conversation, thus revealing their short-term effect on the conversation
as a whole. Additionally, the linkage between increased interest in a topic and the increase of
political polarisation is explored. Our findings reveal that as the interest in the controversial topic
increases, so does the political polarisation.  

URN: http://urn.fi/URN:NBN:fi:hulib-202009294150

