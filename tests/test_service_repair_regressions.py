from __future__ import annotations
import asyncio
import json
import sys
import time
import zipfile
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'qwen_tts_service'))
from document_projects import DocumentProjectManager, extract_document_text
from qwen_protocol import StreamingRequest
from streaming_jobs import StreamingJobManager
from storage_policy import StoragePolicy
from test_macos_qwen_service import load_app_module


def make_app(tmp_path, **kwargs):
    module = load_app_module()
    return module.create_app(output_dir=tmp_path/'output', upload_dir=tmp_path/'uploads',
        preset_dir=tmp_path/'presets', document_project_dir=tmp_path/'books',
        service_job_dir=tmp_path/'jobs', performance_tuning_path=tmp_path/'performance.json',
        preload=False, stt_preload=False, stt_enabled=False, **kwargs)


def test_unicode_password_login_and_rejected_auth_are_not_server_errors(tmp_path):
    app = make_app(tmp_path, access_password='修复测试密码🔒')
    with TestClient(app) as client:
        assert client.get('/api/runtime').status_code == 401
        assert client.post('/login', data={'password':'错误密码'}).status_code == 401
        assert client.get('/api/runtime', headers=[(b'X-API-Key', '修复测试密码🔒'.encode())]).status_code == 200
        response = client.post('/login', data={'password':'修复测试密码🔒'}, follow_redirects=False)
        assert response.status_code == 303
        assert client.get('/api/runtime').status_code == 200
        health = client.get('/api/health').json()
        assert health['service']['api_version'] == 2
        assert len(health['service']['build_id']) == 64


def test_restored_finished_result_replays_and_global_stop_revokes_old_epoch(tmp_path):
    wav = tmp_path/'output'/'generated.wav'
    wav.parent.mkdir(); wav.write_bytes(b'wave-data')
    jobs = StreamingJobManager(tmp_path/'jobs', playback_epoch=1)
    job = jobs.create(summary={'playback_epoch':1})
    job.set_result({'audio_path':str(wav), 'metadata':{'duration_seconds':1}})
    job.update(state='finished', closed=True)
    job.audio_queue.close()
    with TestClient(make_app(tmp_path)) as client:
        result = f'/api/generate-stream/{job.job_id}/result-audio?playback=1'
        initial=client.get(result)
        assert initial.status_code == 200
        previous_epoch=initial.headers['x-playback-epoch']
        assert client.post('/api/service/stop-all').status_code == 200
        assert client.get(result+f"&expected_playback_epoch={previous_epoch}").status_code == 409
        assert client.get(result).status_code == 200
        assert client.get(f'/api/generate-stream/{job.job_id}/result-audio?playback=0').status_code == 200
        assert client.get(f'/api/generate-stream/{job.job_id}/audio').status_code == 410


def test_cached_job_eviction_keeps_result_and_does_not_refresh_revoked_epoch(tmp_path, monkeypatch):
    monkeypatch.setenv('QWEN_TTS_JOB_CACHE_LIMIT','1')
    manager = StreamingJobManager(tmp_path/'jobs', playback_epoch=10)
    first = manager.create(summary={'playback_epoch':10})
    audio = tmp_path/'user-result.wav';audio.write_bytes(b'keep')
    first.set_result({'audio_path':str(audio)});first.update(state='finished')
    first.audio_queue.put_nowait(b'pcm')
    second = manager.create();second.update(state='finished')
    manager.maintenance(now=time.time()+400)
    assert len(manager._jobs) == 1
    assert audio.read_bytes() == b'keep'
    restored = manager.get(first.job_id)
    assert restored.snapshot()['playback_epoch'] == 10
    assert restored.snapshot()['stream_expired']
    assert restored.audio_queue.get_nowait() is None


def test_storage_capacity_refuses_new_data_without_deleting_saved_files(tmp_path, monkeypatch):
    monkeypatch.setenv('QWEN_TTS_STORAGE_MAX_BYTES','4')
    audio = tmp_path/'saved.wav';audio.write_bytes(b'keep-me')
    policy = StoragePolicy((tmp_path,))
    with pytest.raises(Exception) as error:
        policy.check_admission()
    assert error.value.status_code == 507
    assert audio.read_bytes() == b'keep-me'


def test_docx_limits_decompressed_xml_and_rejects_invalid_containers(monkeypatch):
    import document_projects
    monkeypatch.setattr(document_projects,'MAX_DOCUMENT_BYTES',1024)
    output=BytesIO()
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('word/document.xml', b'x'*2048)
    assert len(output.getvalue()) < 1024
    with pytest.raises(ValueError,match='解压'):
        extract_document_text('test.docx',output.getvalue())
    with pytest.raises(ValueError,match='损坏'):
        extract_document_text('test.docx',b'not-zip')
    output=BytesIO()
    with zipfile.ZipFile(output,'w') as archive:
        archive.writestr('word/document.xml','<!DOCTYPE x [<!ENTITY e "expanded">]><x>&e;</x>'.encode('utf-16'))
    with pytest.raises(ValueError,match='实体'):
        extract_document_text('test.docx',output.getvalue())


def manager_for(tmp_path):
    return DocumentProjectManager(root_dir=tmp_path/'books',runtime_session=lambda *_a,**_k:nullcontext(object()),
        synthesize_fn=lambda *_a,**_k:iter(()),request_cls=StreamingRequest,
        generation_lock=nullcontext(),ffmpeg_path='ffmpeg',synthesis_workers=1)


def test_playback_offsets_survive_reload_and_late_updates_do_not_overwrite(tmp_path):
    manager=manager_for(tmp_path)
    project=manager.create_project(name='断点',filename='test.txt',data='第一段。第二段。'.encode(),settings={})
    stamp=time.time()*1000
    manager.update_playback(project['id'],segment_index=0,offset_seconds=7.25,captured_at=stamp,seed=2345)
    manager.update_playback(project['id'],segment_index=0,offset_seconds=0,captured_at=stamp-1)
    restored=manager_for(tmp_path).get_project(project['id'])['playback']
    assert restored['offset_seconds'] == 7.25
    assert restored['seed'] == 2345
    with pytest.raises(ValueError):
        manager.update_playback(project['id'],segment_index=0,offset_seconds=float('nan'))


def test_document_parsing_does_not_block_health_requests(tmp_path, monkeypatch):
    app=make_app(tmp_path)
    original=DocumentProjectManager.create_project
    import threading
    parsing=threading.Event()
    def slow_create(self, **kwargs):
        parsing.set();time.sleep(.4);return original(self,**kwargs)
    monkeypatch.setattr(DocumentProjectManager,'create_project',slow_create)
    with TestClient(app) as client:
        responses=[]
        thread=threading.Thread(target=lambda:responses.append(client.post('/api/document-projects',
            files={'document':('test.txt','普通文本。'.encode(),'text/plain')})))
        thread.start();assert parsing.wait(2)
        began=time.monotonic();assert client.get('/api/health').status_code == 200
        assert time.monotonic()-began < .2
        thread.join(2);assert responses[0].status_code == 200


def test_edit_and_new_settings_invalidate_saved_audio_position(tmp_path):
    manager=manager_for(tmp_path)
    project=manager.create_project(name='重置断点',filename='test.txt',data='这是测试。'.encode(),settings={})
    stamp=time.time()*1000
    manager.update_playback(project['id'],segment_index=0,offset_seconds=5,captured_at=stamp,seed=1234)
    edited=manager.update_segment_text(project['id'],segment_index=0,text='这是改过的测试。')
    assert edited['playback']['offset_seconds']==0
    assert edited['playback']['captured_at']>=stamp
    reset=manager.reset_for_settings(project['id'],{'model_profile':'qwen_1_7b','seed':4321})
    assert reset['playback']['offset_seconds']==0
    assert reset['playback']['seed'] is None


def test_pcm_ttl_keeps_an_active_reader_and_expires_only_after_detachment(tmp_path):
    manager=StreamingJobManager(tmp_path/'jobs')
    job=manager.create();job.update(state='finished')
    job.audio_queue.put_nowait(b'pcm')
    job.active_audio_readers=1
    manager.maintenance(now=time.time()+1000)
    assert not job.snapshot().get('stream_expired')
    assert job.audio_queue.get_nowait()==b'pcm'
    job.audio_queue.put_nowait(b'remainder')
    job.active_audio_readers=0
    manager.maintenance(now=time.time()+1000)
    assert job.snapshot()['stream_expired']
    assert job.audio_queue.get_nowait() is None
