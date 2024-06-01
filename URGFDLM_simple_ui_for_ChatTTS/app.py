import torch
import ChatTTS
from kivy.app import App
from kivy.core.audio import SoundLoader
from kivy.core.text import LabelBase
from kivy.lang import Builder
from kivy.uix.button import Button
import re
LabelBase.register(name='msyh', fn_regular='C:/Windows/Fonts/msyh.ttc')
import soundfile as sf
import tempfile
import numpy as np
import os

os.makedirs('generated_audios', exist_ok=True)

seeds = {
    "中文旁白男性1": {"seed": 48},
    "中文旁白男性2": {"seed": 50},
    "成熟女性": {"seed": 9},
    "中文中年女性": {"seed": 25},
    "中文年轻可爱女性": {"seed": 16},
    "中文年轻活泼女性": {"seed": 51},
    "中文温柔中年女性": {"seed": 41},
    "中文香港口音年轻女性":{"seed":26},
    "中文磁性中年男性": {"seed": 52},
    "中文年轻男性": {"seed": 37},
}
kivy_ui_struct = """
BoxLayout:
    orientation: 'horizontal'
    BoxLayout:
        orientation: 'vertical'
        TextInput:
            id: custom_seed_input
            size_hint: 1, 0.1
            multiline: False
            hint_text: 'Enter custom seed'
        Spinner:
            id: speaker_spinner
            text: 'Choose Speaker Type'
            values: ['成熟女性', '中年女性', '年轻女性', '中年男性', '年轻男性']
            size_hint: 1, 0.1
            font_name: 'msyh'
        TextInput:
            id: input_text
            size_hint: 1, 0.7
            multiline: False
            font_name: 'msyh'
        BoxLayout:
            orientation: 'horizontal'
            size_hint: 1, 0.1
            BoxLayout:
                orientation: 'vertical'
                Label:
                    text: 'Skip Refine Text'
                Switch:
                    id: switch1
                    active: False
            BoxLayout:
                orientation: 'vertical'
                Label:
                    text: 'use Decoder'
                Switch:
                    id: switch2
                    active: True
            BoxLayout:
                orientation: 'vertical'
                Label:
                    text: 'haha'
                Switch:
                    id: switch3
                    active: False
        Button:
            id: submit_button
            text: 'Generate'
            size_hint: 1, 0.1
        BoxLayout:
            orientation: 'vertical'
            size_hint: 1, 0.1
            Label:
                text: '每句最大字数'+ str(int(max_word_slider.value))
                font_name: 'msyh'
            Slider:
                id: max_word_slider
                min: 5
                max: 50
                value: 20
                step: 1
                orientation: 'horizontal'
                on_value: app.set_max_word(self.value)

    ScrollView:
        size_hint: 0.3, 1
        BoxLayout:
            id: audio_list
            orientation: 'vertical'
            size_hint_y: None
            height: self.minimum_height
            width: self.width

"""

class ChatApp(App):
    def build(self):
        self.chat = ChatTTS.Chat()
        self.chat.load_models(source='local', local_path='models')
        self.std, self.mean = torch.load('models/asset/spk_stat.pt').chunk(2)
        self.rnd_spk_emb = None
        self.skip_refine_text = False
        self.use_decoder = True
        self.using_seed = 0
        self.max_word=30
        layout = Builder.load_string(kivy_ui_struct)
        
        # 绑定UI元素
        self.custom_seed_input = layout.ids.custom_seed_input
        self.speaker_spinner = layout.ids.speaker_spinner
        self.input_text = layout.ids.input_text
        self.submit_button = layout.ids.submit_button
        self.switch1 = layout.ids.switch1
        self.switch2 = layout.ids.switch2
        self.audio_list = layout.ids.audio_list
        self.max_word_slider = layout.ids.max_word_slider

        # 设置参数
        self.max_word_slider.value = self.max_word
        
        # 绑定事件
        self.speaker_spinner.bind(text=self.on_speaker_select)
        self.submit_button.bind(on_press=self.infer_and_play)
        self.switch1.bind(active=self.toggle_skip_refine_text)
        self.switch2.bind(active=self.toggle_use_decoder)
        
        return layout
    
    def set_max_word(self, value):
        self.max_word = value
    
    def toggle_skip_refine_text(self, instance, value):
        self.skip_refine_text = value
        
    def toggle_use_decoder(self, instance, value):
        self.use_decoder = value
        
    def on_speaker_select(self, spinner, text):
        seed = seeds[text]["seed"]
        self.choose_speaker(seed)
    
    def choose_speaker(self, seed=0):
        self.deterministic(seed)
        self.rnd_spk_emb = self.chat.sample_random_speaker()

    def infer_and_play(self, instance=None):
        # 如果用户自定义seed，则使用该seed
        custom_seed = self.custom_seed_input.text.strip()
        if custom_seed.isdigit():
            self.choose_speaker(int(custom_seed))

        text = self.input_text.text
        
        if len(text) > self.max_word:
            # 按标点符号分句
            sentences = re.split(r'(?<=[。！？;；.!?;,，])', text)
            sentences = [s.strip() for s in sentences if s.strip() != '']

            # 合并过短的句子接近max_word个字符
            combined_sentences = []
            temp_sentence = ""
            for sentence in sentences:
                if len(temp_sentence) + len(sentence) < self.max_word:
                    temp_sentence += sentence
                else:
                    if temp_sentence:
                        combined_sentences.append(temp_sentence)
                    temp_sentence = sentence
            if temp_sentence:  # 添加最后一个句子，如果有
                combined_sentences.append(temp_sentence)

            sentences = combined_sentences
        else:
            sentences = [text]
        combined_wav = []
        for sentence in sentences:
            params_infer_code = {
                'spk_emb': self.rnd_spk_emb,
            }
            wav = self.chat.infer(sentence,
                                params_infer_code=params_infer_code,
                                use_decoder=self.use_decoder,
                                skip_refine_text=self.skip_refine_text)[0][0]
            combined_wav.append(wav)

        # 合并音频
        if len(combined_wav) > 1:
            combined_wav = np.concatenate(combined_wav)
        else:
            combined_wav = combined_wav[0]
        safe_text = re.sub(r'[^\w]', '', self.input_text.text[:25])
        temp_audio_file = tempfile.NamedTemporaryFile(dir='generated_audios', delete=False, suffix=".wav", prefix="seed"+str(self.using_seed)+"_"+safe_text+"_")
        sf.write(temp_audio_file, combined_wav, 24000, format='WAV', subtype='PCM_24')
        temp_audio_file.close()
        self.play_audio(temp_audio_file.name)
        button_text = "seed:{}_".format(self.using_seed) + self.input_text.text[:10] + "..."
        self.add_audio_to_list(button_text, temp_audio_file.name)  # 添加音频到列表

    def play_audio(self, filename):
        sound = SoundLoader.load(filename)
        if sound:
            sound.play()

    def add_audio_to_list(self, button_text, filename):
        btn = Button(text=button_text, size_hint_y=None, height=50, width=200, font_name='msyh')
        btn.bind(on_press=lambda instance: self.play_audio(filename))
        self.audio_list.add_widget(btn)
        
    def deterministic(self, seed=0):
        self.using_seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    ChatApp().run()
