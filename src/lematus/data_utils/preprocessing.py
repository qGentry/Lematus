import conllu
import re


allowed_char_re = re.compile(r'[^a-zA-Zа-яА-Я.,?!ёЁ"]')
is_russian_word = re.compile(r'[а-яА-ЯёЁ]+')
is_english_word = re.compile(r'[a-zA-Z]+')


class Preprocessor:

    def __init__(self, lang: str):
        if lang == 'rus':
            self.word_checking_re = is_russian_word
        elif lang == 'eng':
            self.word_checking_re = is_english_word
        else:
            raise ValueError(f"Language {lang} is not supported!")

    def get_dataset_from_file(self, data_path, window_size):
        with open(data_path, 'r') as f:
            raw_dataset = f.read()
        sentences = conllu.parse(raw_dataset)
        dataset = self.get_dataset(sentences, window_size)
        return dataset

    def get_dataset(self, sentences, window_size):
        dataset = []
        for sentence in sentences:
            text, lemmas = self.prepare_sentence(sentence)
            for lemma in lemmas:
                target, contexts = self.get_training_samples(text, lemma, window_size)
                dataset.append({
                    "input": contexts[0] + target + contexts[1],
                    "target": list(lemma["lemma"])
                })
        return dataset

    @staticmethod
    def preprocess(text):
        text = allowed_char_re.sub(' ', text)
        text = re.sub(' +', ' ', text)
        return text

    def is_fits_language(self, text):
        return self.word_checking_re.fullmatch(text)

    def prepare_sentence(self, sentence_tokens):
        text = ''
        cur_pointer = 0
        lemmas = []
        for s in sentence_tokens:
            if self.is_fits_language(s['form']):
                lemmas.append(
                    {
                        "form": s['form'],
                        "lemma": s['lemma'],
                        "start_idx": cur_pointer,
                        "end_idx": cur_pointer + len(s['form'])
                    }
                )
            if s['misc'] is not None and s['misc']['SpaceAfter'] == 'No':
                cur_pointer += len(s['form'])
                text += s['form']
            else:
                cur_pointer += len(s['form']) + 1
                text += s['form'] + ' '

        text = text.strip(' ')
        return text, lemmas

    @staticmethod
    def get_training_samples(text, lemma, window_size):
        chars = list(text)
        target_word = lemma['form']

        left_border = max(0, lemma['start_idx'] - window_size)
        left_context = chars[left_border:lemma['start_idx']]

        right_border = min(len(text), lemma['end_idx'] + window_size)
        right_context = chars[lemma['end_idx']:right_border]

        target = ['<lc>'] + list(target_word) + ['<rc>']
        return target, (left_context, right_context)
