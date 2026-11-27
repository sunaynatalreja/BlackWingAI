import pytest
from blackwing.rag import small_context


@pytest.mark.parametrize('output',[
    list(range(230)),
    {'input_ids':list(range(230)), 'attention_mask':[1]*230},
    {'input_ids':[list(range(230))], 'attention_mask':[[1]*230]},
])
def test_real_token_ids_are_counted_across_tokenizer_return_formats(monkeypatch,output):
    class Tokenizer:
        def apply_chat_template(self,*args,**kwargs):return output
    monkeypatch.setattr(small_context,'_tokenizer',lambda model:Tokenizer())
    assert small_context.count_input_tokens([{'role':'user','content':'test'}],strict=True)==230


@pytest.mark.parametrize('output',[{}, {'attention_mask':[1]}, [], [[1,2],[3,4]], ['not','ids']])
def test_unknown_or_multi_conversation_tokenizer_shapes_fail_closed(monkeypatch,output):
    class Tokenizer:
        def apply_chat_template(self,*args,**kwargs):return output
    monkeypatch.setattr(small_context,'_tokenizer',lambda model:Tokenizer())
    with pytest.raises(ValueError,match='tokenizer'):
        small_context.count_input_tokens([{'role':'user','content':'test'}],strict=True)


def test_baseline_also_blocks_overbudget_batchencoding(monkeypatch):
    from blackwing.rag import prompt_baseline
    class Tokenizer:
        def apply_chat_template(self,*args,**kwargs):return {'input_ids':[1]*8000,'attention_mask':[1]*8000}
    monkeypatch.setattr(small_context,'_tokenizer',lambda model:Tokenizer())
    with pytest.raises(ValueError,match='token_budget'):
        prompt_baseline._tokens_preflight([{'role':'user','content':'test'}],768)
