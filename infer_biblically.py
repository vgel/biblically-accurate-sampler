import dataclasses, nltk, re, sys, torch, tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


@dataclasses.dataclass
class TokenTrie:
    nodes: dict[int, "TokenTrie"] = dataclasses.field(default_factory=dict)
    has_leading_space: bool = False
    is_terminal: bool = False

    def add_verse(self, verse: str, words: str, tokenizer) -> None:
        parts = [verse]
        for w in set(words):
            parts.extend((w, " " + w))
        for p in parts:
            tokens = tokenizer.encode(p, add_special_tokens=False)
            cur = self
            for t in tokens:
                cur = cur.nodes.setdefault(t, TokenTrie())
            cur.is_terminal = True
            if p.startswith(" "):
                self.nodes[tokens[0]].has_leading_space = True

    def get(self, token: int) -> "TokenTrie | None":
        return self.nodes.get(token)

    def is_leading_space_tok(self, token: int) -> bool:
        return token in self.nodes and self.nodes[token].has_leading_space


def tokens_that_can_end_word(tokenizer) -> set[int]:
    can_end_word, allowed_chars = set(), set(".!?()")
    for t in tokenizer.vocab.values():
        s = tokenizer.decode([t])  # bc of whitespace handling
        if set(s.strip()) < allowed_chars:
            can_end_word.add(t)
    return can_end_word


@torch.inference_mode
def generate(
    model,
    tokenizer,
    inp: list[int],
    max_new_tokens: int,
    token_trie: TokenTrie,
    can_end_word: set[int],
):
    tokens = torch.tensor([inp[:]]).to(model.device)
    kv = DynamicCache()
    tt_ptr = token_trie
    for _ in range(max_new_tokens):
        out = model(input_ids=tokens, past_key_values=kv, use_cache=True)
        logits = out.logits[:, -1, :].squeeze().cpu()
        options = [t.item() for t in torch.argsort(logits, descending=True)]
        kv = out.past_key_values
        for token in options:
            if tt_ptr.get(token):
                tt_ptr = tt_ptr.get(token)
                break
            elif tt_ptr.is_terminal and token_trie.is_leading_space_tok(token):
                tt_ptr = token_trie.get(token)
                break
            elif tt_ptr.is_terminal and token in can_end_word:
                tt_ptr = token_trie
                break
            elif tt_ptr.is_terminal and token in model.config.eos_token_id:
                return
        tokens = torch.tensor([[token]]).to(tokens.device)
        yield tokenizer.decode([token])


def kjv_verses(path):
    line_re = re.compile(r"(.+? \d+:\d+)(.*)")
    with open(path) as f:
        for line in f:
            verse, text = line_re.match(line).groups()
            yield verse, text, nltk.word_tokenize(text)


if __name__ == "__main__":
    model_name, src_file = sys.argv[1], sys.argv[2]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    token_trie = TokenTrie()
    verses = list(kjv_verses(src_file))
    for verse, _, words in tqdm.tqdm(verses):
        token_trie.add_verse(verse, words, tokenizer)
    can_end_word = tokens_that_can_end_word(tokenizer)

    messages = []
    try:
        while True:
            q = input("> ")
            messages.append({"role": "user", "content": q.strip()})
            inp = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            response = ""
            for p in generate(model, tokenizer, inp, 64, token_trie, can_end_word):
                response += p
                print(p, end="", flush=True)
            print()
            messages.append({"role": "assistant", "content": response})
    except (KeyboardInterrupt, EOFError):
        pass
