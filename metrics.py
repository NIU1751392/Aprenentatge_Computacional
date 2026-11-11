from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import ndcg_score

def spearman_rank_correlation(true_ranking, predicted_ranking):
    correlation, _ = spearmanr(true_ranking, predicted_ranking)
    return correlation

def kendall_tau(y_true, y_pred):
    correlation, _ = kendalltau(y_true, y_pred)
    return correlation

def ndcg(y_true, y_pred): 
    #El dataset funciona al revés, posició 1 és millor que posició 100; ho canviem:
    max_rank = max(y_true)
    relevance = 1 - (y_true - 1) / (max_rank - 1)

    predicted_scores = -y_pred

    score = ndcg_score(
        [relevance],
        [predicted_scores]
    )

    return score

def evaluate_ranking(y_true, y_pred):
    return {
        "Spearman": spearman_rank_correlation(y_true, y_pred),
        "Kendall": kendall_tau(y_true, y_pred),
        "NDCG": ndcg(y_true, y_pred)
    }