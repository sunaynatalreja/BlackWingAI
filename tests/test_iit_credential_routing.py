"""Offline proof that hosted chat and BGE never exchange credentials."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import numpy as np

from blackwing.agents import runtime
from blackwing.rag.embeddings import make_embeddings


def test_chat_and_embedding_use_independent_bearer_keys(monkeypatch,tmp_path):
    lm_key='OFFLINE-LM-ONLY-TEST'
    embedding_key='OFFLINE-EMBEDDING-ONLY-TEST'
    seen=[]
    def chat_handler(request):
        assert request.url.port==1031
        assert request.headers['Authorization']=='Bearer '+lm_key
        assert embedding_key not in request.headers['Authorization']
        seen.append('chat')
        return httpx.Response(200,json={'choices':[]})
    monkeypatch.setattr(runtime,'_LAST_HOSTED_REQUEST_AT',0.0)
    async def chat():
        async with httpx.AsyncClient(transport=httpx.MockTransport(chat_handler)) as client:
            response=await runtime._hosted_post(client,url='https://offline.invalid:1031/v1/chat/completions',
                headers={'Authorization':'Bearer '+lm_key},payload={'max_tokens':50},min_interval_seconds=1.1)
            assert response.status_code==200
    asyncio.run(chat())
    cert=tmp_path/'embedding-ca.crt';cert.write_text('offline fixture; no TLS/network used')
    def embedding_handler(request):
        assert request.url.port==1035 and request.url.path=='/v1/embeddings'
        assert request.headers['Authorization']=='Bearer '+embedding_key
        assert lm_key not in request.headers['Authorization']
        payload=json.loads(request.content)
        seen.append('embedding')
        return httpx.Response(200,json={'data':[{'index':i,'embedding':[1.0]*384} for i,_ in enumerate(payload['input'])]})
    real_client=httpx.Client
    def embedding_client(**kwargs):
        assert kwargs.get('verify')==str(cert)
        return real_client(transport=httpx.MockTransport(embedding_handler))
    monkeypatch.setattr(httpx,'Client',embedding_client)
    class Tokenizer:
        verified=True
        def count(self,text):return 1
    settings=SimpleNamespace(rag_embedding_url='https://offline.invalid:1035/v1',
        rag_embedding_api_key=embedding_key,rag_embedding_ca_cert=str(cert),
        rag_embedding_model='bge-small-en-v1.5',rag_embedding_timeout_seconds=5,
        rag_embedding_retries=0,rag_cache_path='')
    embedder=make_embeddings('iit',tokenizer=Tokenizer(),settings=settings)
    result=embedder.encode(['offline routing test'])
    assert result.shape==(1,384) and np.isfinite(result).all()
    assert seen==['chat','embedding']
