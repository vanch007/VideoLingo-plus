import json, pandas as pd
from pathlib import Path

ws = Path('run_workspaces/e0c803_full_fidelity_rerun_20260926')
stats = json.loads((ws / 'comparison_stats.json').read_text())
eval_data = json.loads((ws / 'output/audio/dubbing_eval.json').read_text())
tasks_df = pd.read_excel(ws / 'output/audio/tts_tasks.xlsx')

s1 = stats['1_source']
s2 = stats['2_prior_dub']
s3 = stats['3_new_dub']

p = []
p.append('<!DOCTYPE html><html><head><meta charset="utf-8"><title>VideoLingo-plus QhmAz
MóOÝw^¦ÍmK[ùqg</title>')
p.append('<style>body{font-family:sans-serif;background:#0f1117;color:#e6edf3;padding:24px;}h1,h2{color:#58a6ff;}.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin-bottom:20px;}.grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;}video{width:100%;border-radius:6px;}table{width:100%;border-collapse:collapse;font-size:13px;}th,td{border:1px solid #30363d;padding:8px;}th{background:#21262d;color:#8b949e;}.val{font-family:monospace;font-weight:bold;color:#79c0ff;}</style></head><body>')
p.append('<h1>Ø<ß­ VideoLingo-plus QhmAz
MóOÝw^¦ÍmKe6b¥TJ</h1>')
p.append('<p>rìzËÐL]åO\S:ÿ<code>' + str(ws.resolve()) + '</code></p>')
p.append('<div class="card"><h2>1. h8_ÃXð[fNrit[ùkÔ (SrG vs eçrH vs e°rH)</h2><div class="grid-3">')
p.append('<div><h3>1. SrG (source.mp4)</h3><video controls src="output/source.mp4"></video><p>eö: <span class="val">' + str(s1['probe']['format']['duration']) + 's</span></p><p>etTTÍ^¦: <span class="val">' + str(s1['loudness']['input_i']) + ' LUFS</span></p><p>zËOSXðRÏ (Side/Mid): <span class="val">' + str(round(s1['side_mid_energy_db'], 2)) + ' dB</span></p><p>Çh7s: <span class="val">' + str(s1['probe']['streams'][1]['sample_rate']) + ' Hz (2ch)</span></p></div>')
p.append('<div><h3>2. eçrHbrG (prior dubbing)</h3><video controls src="../../output/AIMó.mp4"></video><p>eö: <span class="val">' + str(s2['probe']['format']['duration']) + 's</span></p><p>etTTÍ^¦: <span class="val">' + str(s2['loudness']['input_i']) + ' LUFS</span></p><p>zËOSXðRÏ (Side/Mid): <span class="val">' + str(round(s2['side_mid_energy_db'], 2)) + ' dB</span> (ÑNNSUXðSXL))</p><p>Çh7s: <span class="val">' + str(s2['probe']['streams'][1]['sample_rate']) + ' Hz (2ch)</span></p></div>')
p.append('<div><h3>3. g,k!OÝwbrG (new fidelity dubbing)</h3><video controls src="output/AIMó.mp4"></video><p>eö: <span class="val">' + str(s3['probe']['format']['duration']) + 's</span> (NnrG]î 0.0004s)</p><p>etTTÍ^¦: <span class="val">' + str(s3['loudness']['input_i']) + ' LUFS</span> (w\ðP< ' + str(s3['loudness']['input_tp']) + ' dBTP)</p><p>zËOSXðRÏ (Side/Mid): <span class="val">' + str(round(s3['side_mid_energy_db'], 2)) + ' dB</span> (`bY
zËOSXðs¯X)</p><p>Çh7s: <span class="val">' + str(s3['probe']['streams'][1]['sample_rate']) + ' Hz (ØOÝw 48kHz AAC)</span></p></div>')
p.append('</div></div>')
p.append('<div class="card"><h2>2. SåMóNXð[fSÑXðwkbÄmK (30 ga)</h2><table><thead><tr><th>#</th><th>Speaker</th><th>eöôz</th><th>N-eSSðÍ</th><th>ñeMóSðÍ</th><th>[mKeö</th><th>R kÔ</th><th>SÑXðwkbPOyû</th><th>\>èXkQE</th><th>ASR_R</th></tr></thead><tbody>')
for r in eval_data['rows']:
    num = r['number']
    spk_series = tasks_df.loc[tasks_df['number'] == num, 'speaker']
    orig_series = tasks_df.loc[tasks_df['number'] == num, 'origin']
    spk = str(spk_series.values[0]) if not spk_series.empty else 'N/A'
    orig = str(orig_series.values[0]) if not orig_series.empty else ''
    txt = r['text']
    dur = str(round(r.get('real_dur', 0), 2)) + 's'
    sf_val = str(round(r.get('speed_factor', 1.0), 3)) + 'x'
    drift_info = 'w: ' + str(round(r.get('acoustic_start_drift', 0), 2)) + 's / kb: ' + str(round(r.get('acoustic_end_drift', 0), 2)) + 's'
    pad_info = str(round(r.get('acoustic_tail_padding', 0), 2)) + 's'
    cs = str(round(r.get('content_score', 0), 2)) if r.get('content_score') is not None else 'N/A'
    p.append('<tr><td>' + str(num) + '</td><td><b>' + spk + '</b></td><td>' + str(round(r['target_start'], 2)) + '-' + str(round(r['target_end'], 2)) + '</td><td>' + orig + '</td><td>' + txt + '</td><td>' + dur + '</td><td>' + sf_val + '</td><td>' + drift_info + '</td><td>' + pad_info + '</td><td>' + cs + '</td></tr>')
p.append('</tbody></table></div></body></html>')
(ws / 'comparison.html').write_text('\n'.join(p), encoding='utf-8')
print('Saved comparison page to:', ws / 'comparison.html')
