import os
import json
import math
import random
import hashlib
import re
import traceback
from datetime import datetime
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
MIXES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mixes')
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MIXES_DIR, exist_ok=True)

VARIATION_NAMES = {
    0: 'BPM Flow',
    1: 'Energy Build',
    2: 'Energy Drop',
    3: 'BPM Climb',
    4: 'Random',
}

# ── helpers ──────────────────────────────────────────────────────────────────

def extract_video_id(url: str) -> str | None:
    patterns = [
        r'(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})',
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def audio_path(track_id: str) -> str:
    return os.path.join(CACHE_DIR, f'{track_id}.mp3')


def meta_path(track_id: str) -> str:
    return os.path.join(CACHE_DIR, f'{track_id}.json')

# ── routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json(force=True)
    url = (data or {}).get('url', '').strip()
    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    video_id = extract_video_id(url)
    if not video_id:
        # fall back: hash the url
        video_id = hashlib.md5(url.encode()).hexdigest()[:11]

    track_id = video_id
    mp3 = audio_path(track_id)
    meta = meta_path(track_id)

    # ── 1. Download if not cached ─────────────────────────────────────────
    if not os.path.exists(mp3):
        try:
            import yt_dlp
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(CACHE_DIR, f'{track_id}.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'quiet': True,
                'no_warnings': True,
                'nocheckcertificate': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get('title', track_id)
                thumbnail = info.get('thumbnail', '')
                duration_sec = info.get('duration', 0)
        except Exception as e:
            return jsonify({'error': f'Download failed: {str(e)}'}), 500
    else:
        # load cached meta
        title = track_id
        thumbnail = ''
        duration_sec = 0
        if os.path.exists(meta):
            try:
                with open(meta) as f:
                    cached = json.load(f)
                return jsonify(cached)
            except Exception:
                pass

    # ── 2. Analyse with librosa ───────────────────────────────────────────
    analysis = {}
    try:
        import librosa
        y, sr = librosa.load(mp3, sr=22050, mono=True, duration=180)

        tempo_arr, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo_arr) if np.ndim(tempo_arr) == 0 else float(tempo_arr[0])

        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

        # First strong beat — skip silent/empty intro
        rms = librosa.feature.rms(y=y)[0]
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
        rms_thresh = float(np.max(rms)) * 0.15
        first_beat_time = 0.0
        for bt in beat_times:
            idx = np.searchsorted(rms_times, bt)
            if idx < len(rms) and rms[idx] >= rms_thresh:
                first_beat_time = round(float(bt), 3)
                break

        energy = float(np.mean(rms))
        energy_normalised = min(1.0, float(energy) * 20)

        # chroma for key estimation
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                      'F#', 'G', 'G#', 'A', 'A#', 'B']
        estimated_key = note_names[int(np.argmax(chroma_mean))]

        # waveform (downsampled to 500 pts)
        n_pts = 500
        hop = max(1, len(y) // n_pts)
        waveform = [float(np.max(np.abs(y[i:i+hop])))
                    for i in range(0, len(y) - hop, hop)][:n_pts]

        # duration
        duration_sec = float(librosa.get_duration(y=y, sr=sr))

        # phrase length ≈ bars (4 beats/bar × tempo)
        bar_duration = 60.0 / tempo * 4 if tempo > 0 else 4.0
        phrase_8bars = bar_duration * 8
        phrase_16bars = bar_duration * 16

        key_bpm_suggestions = [
            round(tempo * 0.5, 1),
            round(tempo, 1),
            round(tempo * 2, 1),
        ]

        # 32-point energy curve for mix planning
        n_seg = 32
        seg_len = max(1, len(rms) // n_seg)
        energy_curve = [float(np.mean(rms[j*seg_len:(j+1)*seg_len]))
                        for j in range(n_seg)]
        ec_max = max(energy_curve) or 1
        energy_curve_norm = [round(v / ec_max, 4) for v in energy_curve]

        # Beat energies — RMS at each beat position
        beat_energies = []
        for bt in beat_times[:64]:
            idx = int(np.searchsorted(rms_times, bt))
            beat_energies.append(round(float(rms[min(idx, len(rms)-1)]) / (ec_max or 1), 4))

        analysis = {
            'bpm': round(tempo, 1),
            'beat_times': beat_times[:64],
            'first_beat': round(first_beat_time, 3),
            'energy': round(energy_normalised, 3),
            'energy_curve': energy_curve_norm,
            'beat_energies': beat_energies,
            'key': estimated_key,
            'key_bpm_suggestions': key_bpm_suggestions,
            'phrase_8bars': round(phrase_8bars, 2),
            'phrase_16bars': round(phrase_16bars, 2),
            'waveform': waveform,
        }
    except Exception as e:
        analysis = {
            'bpm': 120.0,
            'beat_times': [],
            'energy': 0.5,
            'key': 'C',
            'key_bpm_suggestions': [60.0, 120.0, 240.0],
            'phrase_8bars': 16.0,
            'phrase_16bars': 32.0,
            'waveform': [],
            'analysis_error': str(e),
        }

    result = {
        'track_id': track_id,
        'title': title,
        'url': url,
        'thumbnail': thumbnail,
        'duration': round(duration_sec, 2),
        **analysis,
    }

    with open(meta, 'w') as f:
        json.dump(result, f)

    return jsonify(result)


def _ensure_rich_analysis(track: dict) -> dict:
    """Re-analyse track from cached MP3 if new fields are missing (fast: 60s only)."""
    if 'energy_curve' in track and 'first_beat' in track:
        return track
    mp3 = audio_path(track.get('track_id', ''))
    if not os.path.exists(mp3):
        return track
    try:
        import librosa
        y, sr = librosa.load(mp3, sr=22050, mono=True, duration=60)  # fast: first 60s only
        tempo_arr, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo_arr) if np.ndim(tempo_arr) == 0 else float(tempo_arr[0])
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
        rms = librosa.feature.rms(y=y)[0]
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
        rms_thresh = float(np.max(rms)) * 0.15
        first_beat_time = 0.0
        for bt in beat_times:
            idx = np.searchsorted(rms_times, bt)
            if idx < len(rms) and rms[idx] >= rms_thresh:
                first_beat_time = round(float(bt), 3)
                break
        n_seg = 32
        seg_len = max(1, len(rms) // n_seg)
        ec = [float(np.mean(rms[j*seg_len:(j+1)*seg_len])) for j in range(n_seg)]
        ec_max = max(ec) or 1
        ec_norm = [round(v/ec_max, 4) for v in ec]
        be = []
        for bt in beat_times[:64]:
            idx = int(np.searchsorted(rms_times, bt))
            be.append(round(float(rms[min(idx, len(rms)-1)]) / ec_max, 4))
        track = dict(track)
        track['first_beat']    = first_beat_time
        track['energy_curve']  = ec_norm
        track['beat_energies'] = be
        # persist updated meta
        meta = meta_path(track.get('track_id', ''))
        if os.path.exists(meta):
            with open(meta, 'w') as f:
                json.dump(track, f)
    except Exception:
        pass
    return track


def _find_transition_out(track: dict) -> float:
    """Find best exit point: phrase boundary with naturally declining energy."""
    bpm      = float(track.get('bpm', 120))
    duration = float(track.get('duration', 180))
    first    = float(track.get('first_beat', 0))
    bar_dur  = 60.0 / bpm * 4
    phrase16 = bar_dur * 16

    # Play at least 40 s of real music before transitioning out
    earliest = first + max(40.0, phrase16)
    # Leave at least one phrase for the outro crossfade
    latest   = duration - phrase16 * 0.5
    if earliest >= latest:
        return max(duration * 0.6, earliest)

    # All 16-bar phrase boundaries in the valid window
    candidates = []
    t = first + phrase16
    while t <= latest:
        if t >= earliest:
            candidates.append(t)
        t += phrase16

    if not candidates:
        # fall back to 4-bar boundaries
        t = first + bar_dur * 4
        while t <= latest:
            if t >= earliest:
                candidates.append(t)
            t += bar_dur * 4

    if not candidates:
        return round(max(duration * 0.65, earliest), 2)

    ec = track.get('energy_curve', [])
    n  = len(ec)

    def score(t: float) -> float:
        if n < 4:
            # No curve: prefer ~70 % position
            return 1.0 - abs(t / duration - 0.70)
        idx_now  = min(n-1, int((t / duration) * n))
        idx_pre  = max(0, idx_now - 2)     # energy 2 segments before
        idx_post = min(n-1, idx_now + 1)   # energy 1 segment after
        e_now  = ec[idx_now]
        e_pre  = ec[idx_pre]
        e_post = ec[idx_post]
        # prefer: high energy before → lower now (natural drop = good cue point)
        drop_score    = (e_pre - e_now) * 2.0          # reward energy drop
        energy_score  = e_now * 0.5                    # some energy still there
        position_score = 1.0 - abs(t / duration - 0.72) # prefer ~72 % through track
        return drop_score + energy_score + position_score

    best = max(candidates, key=score)
    return round(best, 2)


def _find_entry_point(track: dict) -> float:
    """Best point to enter the track: first strong beat after intro, on a phrase boundary."""
    first   = float(track.get('first_beat', 0))
    bpm     = float(track.get('bpm', 120))
    bar_dur = 60.0 / bpm * 4
    # Snap to nearest 4-bar boundary at or after first_beat
    bars = first / bar_dur
    phrase_start = math.ceil(bars / 4) * 4 * bar_dur
    # Don't skip more than 32 bars
    phrase_start = min(phrase_start, first + bar_dur * 32)
    return round(max(0.0, phrase_start), 2)


def _sort_by_bpm(tracks: list) -> list:
    """Greedy nearest-neighbour BPM ordering — start from median BPM, always pick closest next."""
    if len(tracks) <= 1:
        return tracks
    bpms = sorted(float(t.get('bpm', 120)) for t in tracks)
    median_bpm = bpms[len(bpms) // 2]
    remaining = list(tracks)
    start = min(remaining, key=lambda t: abs(float(t.get('bpm', 120)) - median_bpm))
    remaining.remove(start)
    ordered = [start]
    while remaining:
        last_bpm = float(ordered[-1].get('bpm', 120))
        nxt = min(remaining, key=lambda t: abs(float(t.get('bpm', 120)) - last_bpm))
        remaining.remove(nxt)
        ordered.append(nxt)
    return ordered


def _sort_variation(tracks: list, variation: int) -> list:
    if variation == 0:  # BPM Flow — greedy nearest-neighbour from median BPM
        return _sort_by_bpm(tracks)
    elif variation == 1:  # Energy Build — slowest/quietest first, builds to hype
        return sorted(tracks, key=lambda t: float(t.get('energy', 0.5)))
    elif variation == 2:  # Energy Drop — high energy opener, chill landing
        return sorted(tracks, key=lambda t: float(t.get('energy', 0.5)), reverse=True)
    elif variation == 3:  # BPM Climb — ascending tempo throughout set
        return sorted(tracks, key=lambda t: float(t.get('bpm', 120)))
    else:               # Random — shuffle, anything goes
        shuffled = list(tracks)
        random.shuffle(shuffled)
        return shuffled


@app.route('/api/generate_mix', methods=['POST'])
def generate_mix():
    data      = request.get_json(force=True)
    tracks    = (data or {}).get('tracks', [])
    variation = int((data or {}).get('variation', 0))
    mix_name  = ((data or {}).get('name') or '').strip()

    if len(tracks) < 2:
        return jsonify({'error': 'Need at least 2 tracks'}), 400

    # Ensure all tracks have rich analysis fields
    tracks = [_ensure_rich_analysis(t) for t in tracks]

    # Order tracks based on chosen variation
    tracks = _sort_variation(tracks, variation)

    sequence    = []
    transitions = []
    cursor      = 0.0   # mix timeline cursor (seconds)

    for i, track in enumerate(tracks):
        bpm      = float(track.get('bpm', 120))
        duration = float(track.get('duration', 180))
        energy   = float(track.get('energy', 0.5))
        bar_dur  = 60.0 / bpm * 4

        # Where this track actually starts playing in the mix (skip intro)
        entry_offset = _find_entry_point(track)

        # Where we start fading out of this track (smart phrase boundary)
        if i < len(tracks) - 1:
            trans_out_abs = _find_transition_out(track)
        else:
            trans_out_abs = duration  # last track plays to end

        # Crossfade duration: longer for lower-energy transitions, min 8 s max 24 s
        crossfade_dur = round(min(24.0, max(8.0, 16.0 * (1.2 - energy))), 2)

        # BPM playback rate (for next track)
        if i < len(tracks) - 1:
            next_bpm     = float(tracks[i+1].get('bpm', 120))
            raw_rate     = next_bpm / bpm if bpm > 0 else 1.0
            playback_rate = round(max(0.92, min(1.08, raw_rate)), 4)
            bpm_match_pct = round(100 - abs(playback_rate - 1.0) / 0.08 * 100, 1)
        else:
            playback_rate = 1.0
            bpm_match_pct = 100.0

        # In the mix timeline this track spans from cursor to cursor+(duration-entry_offset)
        play_duration = duration - entry_offset
        mix_start     = cursor
        mix_end       = cursor + play_duration
        # Transition out point in mix timeline
        mix_trans_out = cursor + (trans_out_abs - entry_offset)
        mix_trans_out = min(mix_trans_out, mix_end - crossfade_dur)

        entry = {
            'track_id':      track.get('track_id'),
            'title':         track.get('title', ''),
            'bpm':           bpm,
            'start_time':    round(mix_start, 2),
            'end_time':      round(mix_end, 2),
            'audio_offset':  round(entry_offset, 2),   # skip this many seconds of audio
            'transition_out': round(mix_trans_out, 2),
            'crossfade_out': crossfade_dur,
            'playback_rate': playback_rate,
        }
        sequence.append(entry)

        if i < len(tracks) - 1:
            transitions.append({
                'from_track':           track.get('track_id'),
                'to_track':             tracks[i+1].get('track_id'),
                'start_at':             round(mix_trans_out, 2),
                'crossfade_duration':   crossfade_dur,
                'bpm_match_quality':    bpm_match_pct,
                'playback_rate_applied': playback_rate,
                'from_audio_pos':       round(trans_out_abs, 2),
                'to_audio_pos':         round(_find_entry_point(tracks[i+1]), 2),
            })
            cursor = mix_trans_out   # next track starts at overlap point
        else:
            cursor = mix_end

    var_name = VARIATION_NAMES.get(variation, 'Custom')
    if not mix_name:
        mix_name = f"{var_name} · {datetime.now().strftime('%b %d %Y %H:%M')}"

    mix_id  = f"mix_{int(datetime.now().timestamp())}"
    mix_doc = {
        'id':             mix_id,
        'name':           mix_name,
        'created_at':     datetime.now().isoformat(),
        'variation':      variation,
        'variation_name': var_name,
        'track_count':    len(sequence),
        'total_duration': round(cursor, 2),
        'sequence':       sequence,
        'transitions':    transitions,
        # self-contained track catalogue: includes YouTube URL so mix is playable elsewhere
        'tracks': {
            t['track_id']: {
                'track_id':  t['track_id'],
                'title':     t.get('title', ''),
                'url':       t.get('url', ''),
                'thumbnail': t.get('thumbnail', ''),
                'bpm':       float(t.get('bpm', 120)),
                'duration':  float(t.get('duration', 0)),
                'key':       t.get('key', ''),
                'energy':    float(t.get('energy', 0.5)),
            } for t in tracks
        },
    }

    mix_file = os.path.join(MIXES_DIR, f'{mix_id}.json')
    with open(mix_file, 'w') as fh:
        json.dump(mix_doc, fh, indent=2)

    return jsonify(mix_doc)


@app.route('/api/mixes', methods=['GET'])
def list_mixes():
    mixes = []
    for fname in os.listdir(MIXES_DIR):
        if not fname.endswith('.json'):
            continue
        try:
            with open(os.path.join(MIXES_DIR, fname)) as fh:
                doc = json.load(fh)
            mixes.append({
                'id':             doc.get('id'),
                'name':           doc.get('name'),
                'created_at':     doc.get('created_at'),
                'variation':      doc.get('variation', 0),
                'variation_name': doc.get('variation_name', ''),
                'track_count':    doc.get('track_count', 0),
                'total_duration': doc.get('total_duration', 0),
            })
        except Exception:
            pass
    mixes.sort(key=lambda m: m.get('created_at', ''), reverse=True)
    return jsonify(mixes)


@app.route('/api/mixes/<mix_id>', methods=['GET'])
def get_mix(mix_id):
    mix_id = re.sub(r'[^A-Za-z0-9_-]', '', mix_id)
    path   = os.path.join(MIXES_DIR, f'{mix_id}.json')
    if not os.path.exists(path):
        return jsonify({'error': 'Mix not found'}), 404
    with open(path) as fh:
        return jsonify(json.load(fh))


@app.route('/api/mixes/<mix_id>', methods=['PATCH'])
def rename_mix(mix_id):
    mix_id = re.sub(r'[^A-Za-z0-9_-]', '', mix_id)
    path   = os.path.join(MIXES_DIR, f'{mix_id}.json')
    if not os.path.exists(path):
        return jsonify({'error': 'Mix not found'}), 404
    name = ((request.get_json(force=True) or {}).get('name') or '').strip()
    if not name:
        return jsonify({'error': 'Name required'}), 400
    with open(path) as fh:
        doc = json.load(fh)
    doc['name'] = name
    with open(path, 'w') as fh:
        json.dump(doc, fh, indent=2)
    return jsonify({'ok': True})


@app.route('/api/mixes/<mix_id>', methods=['DELETE'])
def delete_mix(mix_id):
    mix_id = re.sub(r'[^A-Za-z0-9_-]', '', mix_id)
    path   = os.path.join(MIXES_DIR, f'{mix_id}.json')
    if os.path.exists(path):
        os.remove(path)
    return jsonify({'ok': True})


@app.route('/api/upload', methods=['POST'])
def upload_track():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    f = request.files['file']
    if not f.filename:
        return jsonify({'error': 'Empty filename'}), 400

    # Sanitise and build a stable track_id from filename
    safe_name = re.sub(r'[^A-Za-z0-9._-]', '_', f.filename)
    track_id  = hashlib.md5(safe_name.encode()).hexdigest()[:11]
    mp3       = audio_path(track_id)
    meta      = meta_path(track_id)

    # Return cached result if already analyzed
    if os.path.exists(meta) and os.path.exists(mp3):
        try:
            with open(meta) as fh:
                return jsonify(json.load(fh))
        except Exception:
            pass

    # Save uploaded file
    ext = os.path.splitext(f.filename)[1].lower()
    raw_path = os.path.join(CACHE_DIR, f'{track_id}{ext}')
    f.save(raw_path)

    # Convert to MP3 via ffmpeg if needed
    if ext != '.mp3':
        try:
            import subprocess
            subprocess.run(
                ['ffmpeg', '-y', '-i', raw_path, '-q:a', '2', mp3],
                check=True, capture_output=True
            )
            os.remove(raw_path)
        except Exception as e:
            return jsonify({'error': f'Conversion failed: {e}'}), 500
    else:
        os.rename(raw_path, mp3)

    title = os.path.splitext(f.filename)[0]

    # Analyse
    analysis = {}
    try:
        import librosa
        import numpy as np
        y, sr = librosa.load(mp3, sr=22050, mono=True, duration=180)
        tempo_arr, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo_arr) if np.ndim(tempo_arr) == 0 else float(tempo_arr[0])
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
        rms = librosa.feature.rms(y=y)[0]
        energy = float(np.mean(rms))
        energy_normalised = min(1.0, float(energy) * 20)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        estimated_key = note_names[int(np.argmax(chroma_mean))]
        n_pts = 500
        hop = max(1, len(y) // n_pts)
        waveform = [float(np.max(np.abs(y[i:i+hop]))) for i in range(0, len(y)-hop, hop)][:n_pts]
        duration_sec = float(librosa.get_duration(y=y, sr=sr))
        bar_duration = 60.0 / tempo * 4 if tempo > 0 else 4.0
        analysis = {
            'bpm': round(tempo, 1),
            'beat_times': beat_times[:64],
            'energy': round(energy_normalised, 3),
            'key': estimated_key,
            'key_bpm_suggestions': [round(tempo*0.5,1), round(tempo,1), round(tempo*2,1)],
            'phrase_8bars': round(bar_duration*8, 2),
            'phrase_16bars': round(bar_duration*16, 2),
            'waveform': waveform,
            'duration': round(duration_sec, 2),
        }
    except Exception as e:
        analysis = {'bpm': 120.0, 'beat_times': [], 'energy': 0.5, 'key': 'C',
                    'key_bpm_suggestions': [60.0,120.0,240.0], 'phrase_8bars':16.0,
                    'phrase_16bars':32.0, 'waveform': [], 'duration': 0,
                    'analysis_error': str(e)}

    result = {'track_id': track_id, 'title': title, 'url': '', 'thumbnail': '', **analysis}
    with open(meta, 'w') as fh:
        json.dump(result, fh)
    return jsonify(result)


@app.route('/api/tracks')
def list_tracks():
    """Return all cached & analyzed tracks."""
    results = []
    for fname in os.listdir(CACHE_DIR):
        if fname.endswith('.json'):
            try:
                with open(os.path.join(CACHE_DIR, fname)) as f:
                    data = json.load(f)
                # only include if the MP3 also exists
                if os.path.exists(audio_path(data.get('track_id', ''))):
                    results.append(data)
            except Exception:
                pass
    results.sort(key=lambda x: x.get('title', ''))
    return jsonify(results)


@app.route('/api/audio/<track_id>')
def serve_audio(track_id):
    # sanitise
    track_id = re.sub(r'[^A-Za-z0-9_-]', '', track_id)
    return send_from_directory(CACHE_DIR, f'{track_id}.mp3',
                               mimetype='audio/mpeg')


@app.route('/api/tracks/<track_id>', methods=['DELETE'])
def delete_track(track_id):
    track_id = re.sub(r'[^A-Za-z0-9_-]', '', track_id)
    for path in [audio_path(track_id), meta_path(track_id)]:
        if os.path.exists(path):
            os.remove(path)
    return jsonify({'ok': True})


@app.route('/api/waveform/<track_id>')
def serve_waveform(track_id):
    track_id = re.sub(r'[^A-Za-z0-9_-]', '', track_id)
    mp = meta_path(track_id)
    if os.path.exists(mp):
        try:
            with open(mp) as f:
                meta = json.load(f)
            return jsonify({'waveform': meta.get('waveform', []),
                            'bpm': meta.get('bpm', 120)})
        except Exception:
            pass
    return jsonify({'waveform': [], 'bpm': 120})


@app.route('/api/export_mix', methods=['POST'])
def export_mix():
    """Render the generated mix plan to a single MP3 using ffmpeg."""
    import subprocess, tempfile, shutil
    from flask import send_file

    data = request.get_json(force=True)
    sequence    = (data or {}).get('sequence', [])
    transitions = (data or {}).get('transitions', [])
    if not sequence:
        return jsonify({'error': 'No mix plan provided'}), 400

    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        return jsonify({'error': 'ffmpeg not found on this machine'}), 500

    tmp = tempfile.mkdtemp(prefix='djmix_')
    try:
        # ── Step 1: trim + time-stretch each segment to a temp file ──────────
        seg_files = []
        for i, seg in enumerate(sequence):
            tid = re.sub(r'[^A-Za-z0-9_-]', '', seg.get('track_id', ''))
            src = audio_path(tid)
            if not os.path.exists(src):
                return jsonify({'error': f'Audio file missing for {seg.get("title",tid)}'}), 404

            out_seg = os.path.join(tmp, f'seg_{i:03d}.mp3')
            audio_offset  = float(seg.get('audio_offset', 0))
            play_duration = float(seg['end_time']) - float(seg['start_time'])
            rate          = float(seg.get('playback_rate', 1.0))
            # Clamp atempo to [0.5, 2.0] (ffmpeg limitation)
            rate = max(0.5, min(2.0, rate))

            cmd = [
                ffmpeg, '-y',
                '-ss', str(audio_offset),
                '-t',  str(play_duration / rate),   # read slightly more/less to match rate
                '-i',  src,
                '-af', f'atempo={rate}',
                '-b:a', '192k',
                out_seg,
            ]
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode != 0:
                return jsonify({'error': f'ffmpeg trim failed for segment {i}',
                                'detail': r.stderr.decode()[-500:]}), 500
            seg_files.append(out_seg)

        # ── Step 2: chain with acrossfade ────────────────────────────────────
        # Build crossfade duration map: (from_track_id → crossfade_sec)
        xfade_map = {}
        for tr in transitions:
            xfade_map[tr['from_track']] = float(tr.get('crossfade_duration', 8.0))

        if len(seg_files) == 1:
            final_mp3 = seg_files[0]
        else:
            # Build ffmpeg filter_complex for N-way acrossfade chain
            inputs = []
            for f in seg_files:
                inputs += ['-i', f]

            # Each segment except last gets a crossfade with the next
            filter_parts = []
            last_label = '[0]'
            for i in range(len(seg_files) - 1):
                xd = xfade_map.get(sequence[i]['track_id'], 8.0)
                xd = max(1.0, xd)
                next_label = f'[xf{i}]'
                filter_parts.append(
                    f'{last_label}[{i+1}]acrossfade=d={xd}:c1=tri:c2=tri{next_label}'
                )
                last_label = next_label

            filter_str = ';'.join(filter_parts)
            final_mp3  = os.path.join(tmp, 'mix_final.mp3')
            cmd = [ffmpeg, '-y'] + inputs + [
                '-filter_complex', filter_str,
                '-map', last_label,
                '-b:a', '192k',
                final_mp3,
            ]
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode != 0:
                return jsonify({'error': 'ffmpeg crossfade failed',
                                'detail': r.stderr.decode()[-800:]}), 500

        return send_file(
            final_mp3,
            mimetype='audio/mpeg',
            as_attachment=True,
            download_name='dj_mix.mp3',
        )
    except Exception as e:
        return jsonify({'error': str(e), 'detail': traceback.format_exc()[-500:]}), 500
    finally:
        # Clean up temp dir after response is sent (best effort)
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass


if __name__ == '__main__':
    app.run(debug=True, port=8080)
