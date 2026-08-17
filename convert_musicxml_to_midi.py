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

def find_repeat_markers(selected_part):
    """Scan measure barlines for forward/backward repeat markers.

    Returns (markers, complex_found):
      markers       : [{'measure', 'direction', 'end_s'}, ...]
      complex_found : True if ending/volta/D.C./D.S./al Coda detected.
    Our output MIDI uses 120 BPM (tempo=500000) so each quarter = 0.5s;
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
            else:
                rep = getattr(bl, 'repeat', None)
                if rep is not None:
                    reps.extend(rep if isinstance(rep, (list, tuple)) else [rep])
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
                        'end_s': round(m.offset * 0.5, 3),
                    })
            if getattr(bl, 'ending', None):
                complex_found = True
    return markers, complex_found


def build_segments(notes_list, max_verse_number, has_lyrics_for_verse, markers, complex_found):
    """Build expanded playback-order segments from repeat structure.

    No repeat / complex-repeat -> single whole-song segment.
    Simple backward repeat       -> one segment per lyric verse (the "jump back
                                     and sing again" point becomes a boundary).
    """
    pass_duration_s = round(float(sum(qlen for _p, qlen, _l in notes_list)) * 0.5, 3)
    verse_order = [v for v in range(1, max_verse_number + 1) if has_lyrics_for_verse(v)]
    segments = []
    if complex_found or not any(mk['direction'] == 'backward' for mk in markers):
        total_s = pass_duration_s * max(1, len(verse_order))
        segments.append({
            'seg': 1,
            'start_s': 0.0,
            'end_s': total_s,
            'start_ms': 0,
            'end_ms': int(round(total_s * 1000)),
            'lyrics_verse': verse_order[0] if verse_order else 1,
        })
    else:
        for idx, v in enumerate(verse_order):
            st = idx * pass_duration_s
            en = st + pass_duration_s
            segments.append({
                'seg': idx + 1,
                'start_s': st,
                'end_s': en,
                'start_ms': int(round(st * 1000)),
                'end_ms': int(round(en * 1000)),
                'lyrics_verse': v,
            })
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
        
        # 視窗置中
        root.update_idletasks()
        x = (root.winfo_screenwidth() - root.winfo_reqwidth()) // 2
        y = (root.winfo_screenheight() - root.winfo_reqheight()) // 2
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
            
            # 執行核心轉換
            execute_conversion(selected_part, xml_path)

        # 確認按鈕
        btn_confirm = ttk.Button(root, text="開始轉換", command=on_confirm)
        btn_confirm.pack(pady=15)

        root.mainloop()
        
    except Exception as e:
        messagebox.showerror("錯誤", f"讀取檔案失敗：\n{str(e)}")

def execute_conversion(selected_part, original_xml_path):
    try:
        # 只提取選定聲部中的所有音符與休止符（跳過 ChordSymbol 等和聲標記）
        all_elements = selected_part.flatten().notesAndRests
        raw_notes = [e for e in all_elements if isinstance(e, (music21.note.Note, music21.note.Rest))]

        # 合併延音線（tie）：同音高、以 tie 相接的連續音符合併為一個音，時值相加
        # 每個元素存為 (類型['note'/'rest'], 音高或None, 合併時值, 起始音符的lyrics)
        merged_dur = 0.0
        merged_pitch = None
        merged_lyrics = []
        notes_list = []
        def flush():
            nonlocal merged_dur, merged_pitch, merged_lyrics
            if merged_dur > 0:
                notes_list.append((merged_pitch, merged_dur, merged_lyrics))
                merged_dur, merged_pitch, merged_lyrics = 0.0, None, []

        for el in raw_notes:
            if isinstance(el, music21.note.Rest):
                flush()
                notes_list.append((None, el.duration.quarterLength, []))
                continue

            pitch = el.pitch.midi
            tie_type = el.tie.type if el.tie is not None else None
            # 'stop'/'continue' 表示延續前一音；是延續時併入目前累積的音
            if tie_type in ('stop', 'continue') and merged_pitch is not None:
                merged_dur += el.duration.quarterLength
            else:
                flush()
                merged_pitch = pitch
                merged_dur = el.duration.quarterLength
                merged_lyrics = list(el.lyrics) if el.lyrics else []
        flush()

        # 建立全新的 MIDI（charset 設為 utf-8 以支援中文歌詞）
        mid = MidiFile(charset='utf-8')
        track = MidiTrack()
        mid.tracks.append(track)
        
        # 設定 MIDI 預設速度 (BPM 120)
        track.append(MetaMessage('set_tempo', tempo=500000, time=0))
        
# 自動檢測樂譜中實際存在的段落數量（從 lyrics 的 number 屬性判斷）
        max_verse_number = 1
        for _pitch, _qlen, lyrics in notes_list:
            for lyr in lyrics:
                if isinstance(lyr.number, int) and lyr.number > max_verse_number:
                    max_verse_number = lyr.number

        # 檢查每個段落是否有任何歌詞（段落 2 若完全沒有歌詞則跳過）
        def has_lyrics_for_verse(verse_number):
            for _pitch, _qlen, lyrics in notes_list:
                if verse_lyrics_map(lyrics).get(verse_number, ""):
                    return True
            return False


        # --- repeat 時間點分析 ---
        markers, complex_found = find_repeat_markers(selected_part)
        segments = build_segments(
            notes_list, max_verse_number, has_lyrics_for_verse,
            markers, complex_found,
        )
        if complex_found:
            print("偵測到複雜反覆結構 (volta / D.C. / D.S. 等)，此版不分割，整首送往模型。")

        # 內部重複寫入函式
        def append_verse(verse_number):
            for merged in notes_list:
                pitch, qlen, lyrics = merged
                duration_ticks = int(qlen * 480)
                
                # 休止符
                if pitch is None:
                    track.append(Message('note_off', note=0, velocity=0, time=duration_ticks))
                    continue
                
                # 依段落編號取歌詞（已清掉數字/標點/符號）；若該段沒歌詞，則回退到第一段
                lyric_text = verse_lyrics_map(lyrics).get(verse_number, "")
                if not lyric_text and verse_number != 1:
                    lyric_text = verse_lyrics_map(lyrics).get(1, "")

                if lyric_text:
                    track.append(MetaMessage('lyrics', text=lyric_text, time=0))

                track.append(Message('note_on', note=pitch, velocity=64, time=0))
                track.append(Message('note_off', note=pitch, velocity=64, time=duration_ticks))

        # 寫入所有實際有歌詞的段落
        for verse_num in range(1, max_verse_number + 1):
            if has_lyrics_for_verse(verse_num):
                append_verse(verse_number=verse_num)
        
        # 自動產生輸出檔名
        base_path, _ = os.path.splitext(original_xml_path)
        output_midi_path = f"{base_path}_SoulX.mid"
        
        mid.save(output_midi_path)

        # --- 輸出 repeat 對應 JSON ---
        repeat_json_path = f"{base_path}_SoulX.json"
        payload = {
            "source": os.path.basename(original_xml_path),
            "midi": os.path.basename(output_midi_path),
            "bpm": 120,
            "ticks_per_beat": 480,
            "second_per_beat": 0.5,
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
