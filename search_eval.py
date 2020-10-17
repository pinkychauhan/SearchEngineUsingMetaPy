import math
import sys
import time
import metapy
import pytoml

class InL2Ranker(metapy.index.RankingFunction):
    """
    Create a new ranking function in Python that can be used in MeTA.
    """
    def __init__(self, some_param=1.0):
        self.some_param = some_param
        # You *must* call the base class constructor here!
        super(InL2Ranker, self).__init__()

    def score_one(self, sd):
        """
        You need to override this function to return a score for a single term.
        For fields available in the score_data sd object,
        @see https://meta-toolkit.org/doxygen/structmeta_1_1index_1_1score__data.html
        """
        tfn = sd.doc_term_count * (math.log((1 + (sd.avg_dl/sd.doc_size)), 2))

        score = sd.query_term_weight * (tfn/(tfn + self.some_param)) * (math.log((sd.num_docs + 1)/(sd.corpus_term_count + 0.5), 2))

        #return (self.some_param + sd.doc_term_count) / (self.some_param * sd.doc_unique_terms + sd.doc_size)
        return score

def load_ranker(cfg_file):
    """
    Use this function to return the Ranker object to evaluate,
    The parameter to this function, cfg_file, is the path to a
    configuration file used to load the index.
    """

    #0.39824976473548934	0.3599450172327093	0.9163639789738468	0.6976268690276698
    #return metapy.index.OkapiBM25()
    #return metapy.index.OkapiBM25(2.0, 0.70, 500.0);
    return metapy.index.OkapiBM25(1.2, 0.7, 500.0);
    #return InL2Ranker(some_param=3.0);
    #return metapy.index.JelinekMercer(0.72)
    #return metapy.index.DirichletPrior(158)
    #return metapy.index.AbsoluteDiscount(0.7)
    #return metapy.index.PivotedLength()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: {} config.toml".format(sys.argv[0]))
        sys.exit(1)

    cfg = sys.argv[1]
    print('Building or loading index...')
    idx = metapy.index.make_inverted_index(cfg)
    ranker = load_ranker(cfg)
    ev = metapy.index.IREval(cfg)

    with open(cfg, 'r') as fin:
        cfg_d = pytoml.load(fin)

    query_cfg = cfg_d['query-runner']
    if query_cfg is None:
        print("query-runner table needed in {}".format(cfg))
        sys.exit(1)

    start_time = time.time()
    top_k = 10
    query_path = query_cfg.get('query-path', 'queries.txt')
    query_start = query_cfg.get('query-id-start', 0)

    query = metapy.index.Document()
    ndcg = 0.0
    num_queries = 0

    print('Running queries')
    with open(query_path) as query_file:
        for query_num, line in enumerate(query_file):
            query.content(line.strip())
            results = ranker.score(idx, query, top_k)
            ndcg += ev.ndcg(results, query_start + query_num, top_k)
            num_queries+=1
    ndcg= ndcg / num_queries

    print("NDCG@{}: {}".format(top_k, ndcg))
    print("Elapsed: {} seconds".format(round(time.time() - start_time, 4)))
