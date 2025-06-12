import numpy as np
import pickle
from scipy.special import entr
import bisect

CORRECT = 242
NBINS = 243

class Guesser():
    def __init__(self, dir: str, wordlist: list):
        with open(dir + "table", "rb") as f:
            self.arr: np.ndarray = pickle.load(f)
        with open(dir + "wordlist", "rb") as f:
            all_words: list[str] = pickle.load(f)

        # Select the words in the problem word pool
        is_in_wordlist = [False for i in range(len(all_words))]
        for word in wordlist:
            is_in_wordlist[bisect.bisect_left(all_words, word)] = True
        self.arr = self.arr[is_in_wordlist][:, is_in_wordlist]
        self.words = sorted(wordlist)
        self.word_index = np.arange(len(wordlist))
        self.guess_index = None

    def __str__(self) -> str:
        return f"Array shape: {self.arr.shape}\nWord shape: {self.word_index.shape}"
    
    @staticmethod
    def convert_feedback_to_int(feedback: str) -> int:
        ret = 0
        for i in range(5):
            if feedback[i] == 'b':
                ret += 2 * (3 ** i)
            elif feedback[i] == 'y':
                ret += 3 ** i
        return ret

    def find_guess(self) -> str:
        # returns Argmax(entropy)
        ncols = self.arr.shape[1]
        if ncols == 1:
            self.guess_index = self.word_index[0]
            return self.words[self.guess_index]
        counts = np.apply_along_axis(np.bincount, 1, self.arr, minlength=NBINS).astype(float)
        counts /= ncols
        self.guess_index = entr(counts).sum(axis=1).argmax()
        return self.words[self.guess_index]
        
    def update(self, feedback: int):
        # updates arr and words by feedback
        if self.guess_index is None:
            raise ValueError("Given update query before guessing")
        
        flag = (self.arr[self.guess_index] == feedback)
        self.arr = self.arr[:, flag]
        self.word_index = self.word_index[flag]


# testing
if __name__ == "__main__":
    def check(word, guess):
        l = len(guess)
        result = 0
        bits = 0
        done = 0
        for i in range(l):
            if word[i] == guess[i]:
                result += 2*3**i
                bits |= 1<<i
                done |= 1<<i
        for i in range(l):
            if done & 1<<i:
                continue
            for j in range(l):
                if i == j or bits & 1<<j:
                    continue
                if guess[i] == word[j]:
                    result += 3**i
                    bits |= 1<<j
                    break
        return result

    guesser = Guesser("./data/", ["crane", "flame", "slate"])
    guess = guesser.find_guess()
    print(guess)
    guesser.update(check("crane", guess))
    print(guesser)
