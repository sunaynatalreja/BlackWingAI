from blackwing.rag.report_eval import run

def test_gold_anchor_proxy_not_report_accuracy():
    outputs=[{'task_id':'DEV-1','strategy': 'zero_shot','status':'COMPLETED','schema_valid':True,
              'raw_report':'Clearance was missing, but location unknown.'}]
    refs=[{'task_id':'DEV-1','expected_phrases':'clearance|unknown','forbidden_phrases':'collision'}]
    result=run(outputs,refs)
    assert result['strategies']['zero_shot']['mean_gold_anchor_recall_proxy']==1.0
    assert result['independent_report_accuracy']=='NOT VERIFIED'
