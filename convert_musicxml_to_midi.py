import os
import json
import music21
from mido import MidiFile, MidiTrack, MetaMessage, Message
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

def clean_lyric_text(text):
    """將歌詞清乾淨，只保留『文字』（字母），清掉所有數字、標點、空白與其它符號。

    用 str.isalpha() 判斷：中文漢字、英文 A-Z/a-z 等都是 True 會被保留；
    數字、標點、空白、符號都是 False 會被移除。"""
    return ''.join(ch for ch in (text or '') if ch.isalpha())

def verse_lyrics_map(lyrics):
    """把 lyric 物件轉成 {聲部號碼: 歌詞文字}。
    依序填入，不讓『空的延伸音節』覆蓋同段落已有的真正文字
    （修正 music21 把 extend 誤判成同號碼空字串時，真正歌詞被吃掉的情形）。"""
    mapping = {}
    for lyr in lyrics:
        num = lyr.number
        txt = clean_lyric_text(lyr.text)
        if num not in mapping or (txt and not mapping[num]):
            mapping[num] = txt
    return mapping

def find_repeat_markers(selected_part, second_per_beat=0.5):
    """Scan measure barlines for forward/backward repeat markers.

    Returns (markers, complex_found):
      markers       : [{'measure', 'direction', 'end_s'}, ...]
      complex_found : True if ending/volta/D.C./D.S./al Coda detected.
    Our output MIDI uses the detected BPM, so we use second_per_beat.
    measure.offset is in quarterLength units.
    """
    markers = []
    complex_found = False
    try:
        from music21 import repeat as _m21repeat
        if list(selected_part.recurse().getElementsByClass(_m21repeat.RepeatExpression)):
            complex_found = True
    except Exception:
        pass
    for m in selected_part.getElementsByClass(music21.stream.Measure):
        for bl in (getattr(m, 'leftBarline', None), getattr(m, 'rightBarline', None)):
            if bl is None:
                continue
            # music21 現代版本會直接把 bar.Repeat 當成 measure 的
            # leftBarline / rightBarline（direction 為 'start'/'end'）；
            # 舊版本則包在 Barline.repeat 屬性裡。兩種都要支援。
            reps = []
            if isinstance(bl, music21.bar.Repeat):
                reps.append(bl)
            elif hasattr(bl, 'repeat'):
                rep = getattr(bl, 'repeat', None)
                if rep is not None:
                    reps.extend(rep if isinstance(rep, (list, tuple)) else [rep])
            
            # 額外檢查 Measure 是否有 repeat 屬性 (部分 music21 版本)
            if not reps and hasattr(m, 'repeat'):
                rep_attr = getattr(m, 'repeat')
                if rep_attr:
                    reps.append(rep_attr)

            for r in reps:
                d = getattr(r, 'direction', None)
                d = {'start': 'forward', 'end': 'backward',
                     'start-end': 'forward-backward'}.get(d, d)
                if d == 'forward-backward':
                    d = ['forward', 'backward']
                if isinstance(d, str):
                    d = [d]
                for dd in d:
                    markers.append({
                        'measure': int(m.number),
                        'direction': dd,
                        'end_s': round(m.offset * second_per_beat, 3),
                    })
            if getattr(bl, 'ending', None):
                complex_found = True
    return markers, complex_found


def build_segments(notes_list, max_verse_number, has_lyrics_for_verse, markers, complex_found, second_per_beat=0.5):
    """Build expanded playback-order segments from repeat structure.

    No repeat / complex-repeat -> single whole-song segment.
    Simple backward repeat       -> one segment per lyric verse (the "jump back
                                  and sing again" point becomes a boundary).
    """
    verse_order = [v for v in range(1, max_verse_number + 1) if has_lyrics_for_verse(v)]
    segments = []
    backward_repeat_measures = [mk['measure'] for mk in markers if mk['direction'] == 'backward']
    repeat_end_m = max(backward_repeat_measures) if (backward_repeat_measures and not complex_found) else None

    if complex_found or not backward_repeat_measures:
        total_qlen = sum(item[1] for item in notes_list)
        total_s = round(float(total_qlen) * second_per_beat * max(1, len(verse_order)), 3)
        segments.append({
            'seg': 1,
            'start_s': 0.0,
            'end_s': total_s,
            'start_ms': 0,
            'end_ms': int(round(total_s * 1000)),
            'lyrics_verse': verse_order[0] if verse_order else 1,
        })
    else:
        current_time_s = 0.0
        for idx, v in enumerate(verse_order):
            is_last_verse = (idx == len(verse_order) - 1)
            v_qlen = 0.0
            for item in notes_list:
                m_num = item[3] if len(item) > 3 else None
                if (not is_last_verse) and (repeat_end_m is not None) and (m_num is not None) and (m_num > repeat_end_m):
                    continue
                v_qlen += item[1]
            v_dur_s = v_qlen * second_per_beat
            st = current_time_s
            en = st + v_dur_s
            segments.append({
                'seg': idx + 1,
                'start_s': round(st, 3),
                'end_s': round(en, 3),
                'start_ms': int(round(st * 1000)),
                'end_ms': int(round(en * 1000)),
                'lyrics_verse': v,
            })
            current_time_s = en
    return segments

def select_part_and_convert(xml_path):
    try:
        # 1. 解析 MusicXML
        print(f"正在解析樂譜：{os.path.basename(xml_path)}...")
        score = music21.converter.parse(xml_path)
        
        # 2. 獲取樂譜中的所有聲部 (Parts)
        # 格式會是：{ "Part ID 或名稱": music21.stream.Part 物件 }
        parts_dict = {}
        for p in score.parts:
            # 優先拿聲部名稱 (partName)，沒有就拿 ID
            name = p.partName if p.partName else f"Track - {p.id}"
            parts_dict[name] = p
            
        if not parts_dict:
            messagebox.showerror("錯誤", "樂譜中找不到任何有效聲部！")
            return

        # 3. 建立 Tkinter 彈出視窗
        root = tk.Tk()
        root.title("SoulX-Singer MIDI 轉換器")
        root.geometry("400x180")
        root.resizable(False, False)
        
        # 強制視窗置頂，避免被推到後面
        root.attributes('-topmost', True)
        
        # 視窗置中
        root.update_idletasks()
        x = (root.winfo_screenwidth() - root.winfo_reqwidth()) // 2
        y = (root.winfo_screenheight() - root.winfo_reqwidth()) // 2
        root.geometry(f"+{x}+{y}")

        # 介面標籤
        label = ttk.Label(root, text=f"已載入: {os.path.basename(xml_path)}\n請選擇【主旋律歌詞軌】:", font=("Arial", 10))
        label.pack(pady=15)

        # 下拉選單 (Combobox)
        part_names = list(parts_dict.keys())
        combo = ttk.Combobox(root, values=part_names, state="readonly", width=35)
        combo.pack(pady=5)
        combo.current(0) # 預設選第一個

        # 點擊確認後的執行邏輯
        def on_confirm():
            selected_name = combo.get()
            selected_part = parts_dict[selected_name]
            root.destroy() # 關閉視窗
            
            # 嘗試從樂譜中獲取 Tempo
            bpm = 120 # 預設
            try:
                tempo_marks = score.recurse().getElementsByClass(music21.tempo.MetronomeMark)
                if tempo_marks:
                    mark = tempo_marks[0]
                    # 根據 debug 結果，數值存放在 _number 屬性中
                    if hasattr(mark, '_number') and mark._number is not None:
                        bpm = int(mark._number)
                    elif hasattr(mark, 'bpm'):
                        bpm = int(mark.bpm)
                    elif hasattr(mark, 'metronomeMark'):
                        bpm = int(mark.metronomeMark)
            except Exception as e:
                print(f"Tempo 獲取失敗，使用預設 120: {e}")
            
            # 執行核心轉換
            execute_conversion(selected_part, xml_path, bpm)

        # 確認按鈕
        btn_confirm = ttk.Button(root, text="開始轉換", command=on_confirm)
        btn_confirm.pack(pady=15)

        root.mainloop()
        
    except Exception as e:
        messagebox.showerror("錯誤", f"讀取檔案失敗：\n{str(e)}")

def execute_conversion(selected_part, original_xml_path, bpm=120):
    try:
        # 提取所有音符與休止符
        all_elements = selected_part.flatten().notesAndRests
        
        # 建立 notes_list，記錄 (pitch, duration, lyrics, measureNumber)
        notes_list = []
        valid_elements = [e for e in all_elements if isinstance(e, (music21.note.Note, music21.note.Rest)) and e.duration.quarterLength > 0]
        valid_elements.sort(key=lambda e: e.offset)
        
        if not valid_elements:
            notes_list = []
        else:
            merged_pitch = None
            merged_dur = 0.0
            merged_lyrics = []
            merged_m_num = getattr(valid_elements[0], 'measureNumber', None)
            
            for i in range(len(valid_elements)):
                el = valid_elements[i]
                m_num = getattr(el, 'measureNumber', None)
                
                # 判定是否為延音 (tie): 同音高 且 offset 正好銜接
                is_tie = False
                if i > 0:
                    prev = valid_elements[i-1]
                    if isinstance(el, music21.note.Note) and isinstance(prev, music21.note.Note):
                        same_pitch = el.pitch == prev.pitch
                        contiguous = el.offset == prev.offset + prev.duration.quarterLength
                        same_measure = getattr(el, 'measureNumber', None) == getattr(prev, 'measureNumber', None)
                        if same_pitch and contiguous and same_measure:
                            el_tie_type = getattr(el.tie, 'type', None)
                            prev_tie_type = getattr(prev.tie, 'type', None)
                            if el_tie_type in ('stop', 'continue') or prev_tie_type in ('start', 'continue'):
                                is_tie = True        
                                         
                if isinstance(el, music21.note.Rest):
                    # 先 flush 之前的音符
                    if merged_dur > 0:
                        notes_list.append((merged_pitch, merged_dur, merged_lyrics, merged_m_num))
                        merged_dur, merged_pitch, merged_lyrics = 0.0, None, []
                    
                    notes_list.append((None, el.duration.quarterLength, [], m_num))
                    
                elif is_tie:
                    merged_dur += el.duration.quarterLength
                    if el.lyrics:
                        merged_lyrics.extend(el.lyrics)
                else:
                    # Flush 之前的音符
                    if merged_dur > 0:
                        notes_list.append((merged_pitch, merged_dur, merged_lyrics, merged_m_num))
                    
                    merged_pitch = el.pitch.midi
                    merged_dur = el.duration.quarterLength
                    merged_lyrics = list(el.lyrics) if el.lyrics else []
                    merged_m_num = m_num
            
            # 最後一次 flush
            if merged_dur > 0:
                notes_list.append((merged_pitch, merged_dur, merged_lyrics, merged_m_num))

        # 建立全新的 MIDI（charset 設為 utf-8 以支援中文歌詞）
        mid = MidiFile(charset='utf-8')
        track = MidiTrack()
        mid.tracks.append(track)
        
        # 設定 MIDI 速度 (動態 BPM)
        midi_tempo = int(60000000 / bpm)
        track.append(MetaMessage('set_tempo', tempo=midi_tempo, time=0))
        
        # 計算每拍秒數
        second_per_beat = 60.0 / bpm
        
        # 自動檢測樂譜中實際存在的段落數量（從 lyrics 的 number 屬性判斷）
        max_verse_number = 1
        for item in notes_list:
            lyrics = item[2]
            for lyr in lyrics:
                if isinstance(lyr.number, int) and lyr.number > max_verse_number:
                    max_verse_number = lyr.number

        # 檢查每個段落是否有任何歌詞（段落 2 若完全沒有歌詞則跳過）
        def has_lyrics_for_verse(verse_number):
            for item in notes_list:
                lyrics = item[2]
                if verse_lyrics_map(lyrics).get(verse_number, ""):
                    return True
            return False


        # --- repeat 時間點分析 ---
        markers, complex_found = find_repeat_markers(selected_part, second_per_beat=second_per_beat)
        segments = build_segments(
            notes_list, max_verse_number, has_lyrics_for_verse,
            markers, complex_found,
            second_per_beat=second_per_beat,
        )
        if complex_found:
            print("偵測到複雜反覆結構 (volta / D.C. / D.S. 等)，此版不分割，整首送往模型。")

        backward_repeat_measures = [mk['measure'] for mk in markers if mk['direction'] == 'backward']
        repeat_end_m = max(backward_repeat_measures) if (backward_repeat_measures and not complex_found) else None

        # 內部重複寫入函式
        def append_verse(verse_number, is_last_verse):
            for merged in notes_list:
                pitch, qlen, lyrics = merged[0], merged[1], merged[2]
                m_num = merged[3] if len(merged) > 3 else None
                
                # 若非最後一段，且該小節位於反覆結束記號之後，則跳過 (反覆折返)
                if (not is_last_verse) and (repeat_end_m is not None) and (m_num is not None) and (m_num > repeat_end_m):
                    continue

                duration_ticks = int(qlen * 480)
                
                # 休止符
                if pitch is None:
                    track.append(Message('note_off', note=0, velocity=0, time=duration_ticks))
                    continue
                
                # 依段落編號取歌詞（已清掉數字/標點/符號）；若該段沒歌詞，則回退到第一段
                lyric_text = verse_lyrics_map(lyrics).get(verse_number, "")
                if not lyric_text and verse_number != 1:
                    lyric_text = verse_lyrics_map(lyrics).get(1, "")

                # --- 歌詞 MetaMessage 的 time 必須為 0，且必須在 note_on 之前發送 ---
                if lyric_text:
                    track.append(MetaMessage('lyrics', text=lyric_text, time=0))

                track.append(Message('note_on', note=pitch, velocity=64, time=0))
                track.append(Message('note_off', note=pitch, velocity=64, time=duration_ticks))

        # 寫入所有實際有歌詞的段落
        verse_order = [v for v in range(1, max_verse_number + 1) if has_lyrics_for_verse(v)]
        for idx, verse_num in enumerate(verse_order):
            is_last = (idx == len(verse_order) - 1)
            append_verse(verse_number=verse_num, is_last_verse=is_last)
        
        # 自動產生輸出檔名
        base_path, _ = os.path.splitext(original_xml_path)
        output_midi_path = f"{base_path}_SoulX.mid"
        
        mid.save(output_midi_path)

        # --- 輸出 repeat 對應 JSON ---
        repeat_json_path = f"{base_path}_SoulX.json"
        payload = {
            "source": os.path.basename(original_xml_path),
            "midi": os.path.basename(output_midi_path),
            "bpm": bpm,
            "ticks_per_beat": 480,
            "second_per_beat": round(second_per_beat, 3),
            "repeat_markers": markers,
            "complex_repeat_present": complex_found,
            "segments": segments,
        }
        with open(repeat_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # 彈窗提示成功
        root_success = tk.Tk()
        root_success.withdraw()
        messagebox.showinfo(
            "成功",
            f"🎉 MIDI 轉換完成！\n已儲存至：\n{output_midi_path}\n\n"
            f"Repeat 對應檔：\n{repeat_json_path}",
        )
        root_success.destroy()
        
    except Exception as e:
        root_err = tk.Tk()
        root_err.withdraw()
        messagebox.showerror("轉換失敗", f"發生未知錯誤：\n{str(e)}")
        root_err.destroy()

# ==========================================
# 程式進入點：自動打開檔案選擇器
# ==========================================
if __name__ == "__main__":
    # 初始化一個隱藏的主視窗，純粹為了調用檔案選擇器
    main_init = tk.Tk()
    main_init.withdraw()
    
    # 讓使用者選擇 MusicXML 檔案
    file_path = filedialog.askopenfilename(
        title="請選取從 MuseScore 導出的 MusicXML 檔案",
        filetypes=[("MusicXML 檔案", "*.musicxml *.xml"), ("所有檔案", "*.*")]
    )
    
    main_init.destroy()
    
    if file_path:
        select_part_and_convert(file_path)
    else:
        print("未選擇任何檔案，程式結束。")
