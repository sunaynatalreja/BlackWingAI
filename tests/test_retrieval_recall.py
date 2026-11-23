from copy import deepcopy
import pytest
from blackwing.evaluation.retrieval import score_rankings


def assets():
    qrels = {'unit':'chunk','index_signature':'frozen','scope_id':'reviewed',
             'approval':{'status':'APPROVED'},'eligible_ids':['A','B','C','D','E','F'],
             'questions':[{'question_id':'Q1','query':'crossing','complete_relevance':True,'relevant_ids':['A','B']},
                          {'question_id':'Q2','query':'abort','complete_relevance':True,'relevant_ids':['C']}]}
    ranking = {'unit':'chunk','index_signature':'frozen','scope_id':'reviewed','queries':[
        {'question_id':'Q1','method':'bm25','ranked_ids':['D','E','F','A','C','B']},
        {'question_id':'Q2','method':'bm25','ranked_ids':[]}]}
    return qrels,ranking


def test_recall_denominator_is_all_relevant_items_and_zero_hits_stay_in_mean():
    q,r = assets(); original=deepcopy((q,r)); out=score_rankings(q,r)
    assert out['queries'][0]['recall@5']==0.5
    assert out['queries'][0]['recall@10']==1.0
    assert out['queries'][1]['recall@5']==0.0
    assert out['summary'][0]['recall@5']==0.25
    assert out['summary'][0]['recall@10']==0.5
    assert (q,r)==original


@pytest.mark.parametrize('field,value',[('index_signature','other'),('scope_id','other'),('unit','document')])
def test_incompatible_ranking_identity_refused(field,value):
    q,r=assets();r[field]=value
    with pytest.raises(ValueError):score_rankings(q,r)


def test_missing_judgments_and_zero_relevance_are_not_zero_recall():
    q,r=assets();q['questions'][0]['complete_relevance']=False;q['questions'][1]['relevant_ids']=[]
    out=score_rankings(q,r)
    assert out['summary'][0]['recall@5'] is None
    assert out['summary'][0]['queries_measured']==0


def test_duplicate_stale_missing_or_fallback_results_cannot_inflate_recall():
    for change in ['duplicate','stale','missing','fallback']:
        q,r=assets()
        if change=='duplicate':r['queries'][0]['ranked_ids']=['A','A','B']
        if change=='stale':r['queries'][0]['ranked_ids']=['UNKNOWN']
        if change=='missing':r['queries'].pop()
        if change=='fallback':
            r['queries'][0]['actual_method']='fallback'
            out=score_rankings(q,r)
            assert out['queries'][0]['recall@5'] is None
            assert out['queries'][0]['status']=='provider_or_method_failure'
            continue
        with pytest.raises(ValueError):score_rankings(q,r)


def test_document_and_chunk_metrics_cannot_be_silently_interchanged():
    q,r=assets();q['unit']=r['unit']='document'
    assert score_rankings(q,r)['unit']=='document'
    q['approval']['status']='PENDING'
    with pytest.raises(ValueError):score_rankings(q,r)
