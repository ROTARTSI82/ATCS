#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <fstream>

enum TokType : uint8_t {
  ALPH, NUM, SYM, PRESYM, SPACE, NEWLINE, SPECIAL, WORD_START, NUM_START, SYM_START, MIXED
};

std::string toktoa(TokType type) {
  switch (type) {
    case ALPH:
      return "alph";
    case NUM:
      return "num";
    case SYM:
      return "sym";
    case PRESYM:
      return "psym";
    case NEWLINE:
      return "newl";
    case SPACE:
      return "space";
    case SPECIAL:
      return "spec";
    case WORD_START:
      return "wstrt";
    case NUM_START:
      return "nstrt";
    case SYM_START:
      return "sstrt";
    case MIXED:
      return "mix";
    default:
      return "unk";
  }
}

struct Token {
  size_t left, right;
  TokType type;
  
  bool operator==(const Token &rhs) const {
    return left == rhs.left && right == rhs.right;
  }
};

template <>
struct std::hash<Token>
{
  std::size_t operator()(const Token& k) const
  {
    return std::hash<size_t>()(k.right) ^ (std::hash<size_t>()(k.left) << 24);
  }
};

struct Vocabulary {
  std::vector<Token> tokens;
  std::unordered_map<std::string, size_t> tokmap;
  size_t max_toklen = 0;

  std::string resolve(size_t tokid) {
    if (tokid < 256)
      return std::string{static_cast<char>(tokid)};
    if (tokid -256>= tokens.size())
      std::cout << "uh oh on tok = " << tokid << " <| idx= " << tokid-256 << " size=" << tokens.size() << "\n";
    auto &tok = tokens[tokid - 256];
    return resolve(tok.left) + resolve(tok.right);
  }

  TokType resolve_type(size_t tokid) {
    if ('a' <= tokid && tokid <= 'z' || 'A' <= tokid && tokid <= 'Z') 
      return ALPH;
    if ('0' <= tokid && tokid <= '9')
      return NUM;
    if ('!' <= tokid && tokid <= '~') { 
      if (tokid == '_' || tokid == '-')
        return PRESYM;
      return SYM;
    }
    
    if (tokid == ' ')
      return SPACE;
    if (tokid == '\n')
      return NEWLINE;
    if (tokid < 256)
      return SPECIAL;

    if (tokid - 256 < tokens.size())
      return tokens[tokid - 256].type;

    TokType left = resolve_type(tokens[tokid - 256].left);
    TokType right = resolve_type(tokens[tokid - 256].right);
    return resolve_type(left, right);
  }

  TokType resolve_type(TokType left, TokType right) {
    // if (left == SPECIAL || right == SPECIAL)
    //   return SPECIAL;
    if (left == right && (left == ALPH || left == NUM || left == SYM || left == SPECIAL || left == SPACE))
      return left;
    if (left == PRESYM && right == SYM || left == SYM && right == PRESYM || left == PRESYM && right == PRESYM)
      return SYM;
    if (right == ALPH && (left == SPACE || left == WORD_START || left == PRESYM))
      return WORD_START;
    if (right == NUM && (left == SPACE || left == NUM_START))
      return NUM_START;
    if ((right == SYM || right == PRESYM) && (left == SPACE || left == SYM_START))
      return SYM_START;
    
    return MIXED;
  }


  void bpe(size_t *stream, size_t size) {
    size_t *alt = new size_t[size];
    size_t *deletor = alt;

    std::unordered_map<Token, size_t> freqs{};

    for (int vocab_size = 0; vocab_size < 1024; vocab_size++) {
      Token selected{31415, 271828};
      size_t maxocc = 0;
      for (size_t i = 0; i < size - 1; i++) {
        Token t{stream[i], stream[i+1]};
        freqs[t] += 1;

        // we can break the rules if it's common enough
        // if (freqs[t] > 2*maxocc) {
        //   maxocc = freqs[t];
        //   selected = t;
        // }

        TokType right = resolve_type(t.right);
        TokType left = resolve_type(t.left);
        t.type = resolve_type(left, right);
        if (t.type == MIXED)
          continue;
       
        if (freqs[t] > maxocc) {
          maxocc = freqs[t];
          selected = t;
        }
      }

      if (maxocc < 256) {
        break; // we combined all the tokens we can! or it's just not worth it.
      }

      size_t idx = tokens.size() + 256;
      tokens.emplace_back(selected);
      tokmap[resolve(idx)] = idx;

      size_t alt_idx = 0;
      for (size_t i = 0; i < size; i++) {
        if (i < size - 1 && Token{stream[i], stream[i+1]} == selected) {
          alt[alt_idx++] = idx;
          i++;
        } else {
          alt[alt_idx++] = stream[i];
        }
      }

      std::cout << idx << '\t' << size << "->" << alt_idx << "\tocc " << freqs[selected] << "\t" << "\ttok (" << resolve(selected.left) << "|" << resolve(selected.right) << ")\t\t|" << resolve(idx) << "|\t"
        << "\t\t" << toktoa(resolve_type(idx)) << "\t(" << toktoa(resolve_type(selected.left)) << " + " << toktoa(resolve_type(selected.right)) << ")" << std::endl;

      size = alt_idx;
      std::swap(stream, alt);
      freqs.clear();
    }
    

    delete[] deletor;
  }

};

int main() {
  std::ifstream file("dataset.txt", std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<uint8_t> buffer(size);
  if (file.read((char*)buffer.data(), size))
  {
    Vocabulary vocab;
    std::vector<size_t> expanded;
    for (const auto &i : buffer) {
      expanded.emplace_back(static_cast<size_t>(i));
      if (expanded.back() > 256)
        std::cout << "WTF\n";
    }
    
    vocab.bpe(expanded.data(), expanded.size());

    std::ofstream out("vocab.txt", std::ios::binary);
    size_t idx = 256;
    for (auto &tok : vocab.tokens) {
      std::string str = vocab.resolve(idx);
      size_t len = str.length();

      out.write((char*) &len, sizeof(size_t));
      out.write(str.data(), str.length());
      
      idx++;
    }
  }
}
