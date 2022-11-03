import pandas as pd
import random

class NumberToString:
    def __init__(self, path):
        self.number_df = pd.read_excel(path, sheet_name = 'numberparser')
        self.number_df["number_sign"] = self.number_df["number_sign"].apply(self.__change_type)
        self.number_map = {1:"X", 2:"chục" , 3:"trăm", 4:"ngàn|nghìn", 7:"triệu", 10:"tỷ|tỉ"}
        self.extract_num_list = []
        for x in self.number_df.number_sign.to_list():
            if isinstance(x,int):
                self.extract_num_list.append(x)

    def __change_type(self, x):
        if isinstance(x, float) or isinstance(x, int):
            x = int(x)
        elif isinstance(x, str):
            x = x.replace("X", "(\d)")
        return x

    def parse_hundered_number(self, number:int):
        assert len(str(number)) <= 3, \
            "must be a number smaller than 999 interger"
        hundered = number//100
        number = number%100
        dozen = number//10
        interger = number%10
        pre_num, pre_posfix = None, None
        words_list = []
        for i, x in enumerate([hundered, dozen, interger]):
            tn = self.number_df[self.number_df["number_sign"]==x]
            
            if x != 0 and i == 2 and pre_num not in [0,1]:
                tn = random.choice(tn["in_posfix"].to_list()[0].split("|"))
            else:
                tn = random.choice(tn["hand_write"].to_list()[0].split("|"))

            if i == 0:
                posf = "trăm"
                words_list.extend([tn, posf])
            if i == 1:
                if x == 0:
                    if i>=1:
                        posf = random.choice(["lẻ","linh"])
                        words_list.append(posf)
                elif x == 1:
                    posf = "mười"
                    words_list.append(posf)
                else:
                    posf = "mươi"
                    words_list.extend([tn, posf])
            if i == 2:
                if x != 0:
                    words_list.append(tn)
                elif pre_posfix in ["lẻ", "linh"]:
                    words_list = words_list[:-1]
            pre_num, pre_posfix= x, posf
        return words_list

    def parse_number(self, number:int):
        numbe_len = len(str(number))
        if numbe_len <= 10:
            i, dummy_number = 0, {}
            while number != 0:
                x = number%1000
                dummy_number[self.number_map[i*3+1]] = x
                number = number//1000
                i += 1
            return dummy_number    
        else:
            result, numbers = [], []
            while number != 0:
                numbers.append(int(number%1000000000))
                number = int(number//1000000000)
            numbers = list(reversed(numbers))
            for n in numbers:
                result.append(self.parse_number(n))
            return result

    def __call__(self, number:int):
        if len(str(number)) == 1:
            tn = self.number_df[self.number_df["number_sign"]==number]
            return random.choice(tn["hand_write"].to_list()[0].split("|"))
        
        k = self.parse_number(number)
        if not isinstance(k, list):
            k = [k]

        string = ""
        for i, x in enumerate(k):
            keys = list(reversed(x.keys()))
            for j, key in enumerate(keys):
                posf = random.choice(key.split("|")) if key != "X" else ""
                parsed_words = self.parse_hundered_number(x[key])
                if i+j == 0 and " ".join(parsed_words).find("không trăm") != -1:
                    parsed_words = parsed_words[2:]
                    if parsed_words[0] in ["lẻ","linh"]:
                        parsed_words = parsed_words[1:]

                string += " ".join(parsed_words) + " " + posf + " "
            if i+1 != len(k):
                string = string.strip() + " tỷ "
        return string.strip()