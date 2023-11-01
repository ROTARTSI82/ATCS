#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#define READER_ASSERT(expr, msg) if (!(expr)) {             \
    std::cout << "failed: " << #expr ": " << (msg) << '\n'; \
    return false;                                           \
}

#define E2S_CASE(name) case (name): return #name;

#define R1_CASES case CONSTANT_String: case CONSTANT_Class: case CONSTANT_MethodType
#define R2_CASES case CONSTANT_Methodref: case CONSTANT_Fieldref: case CONSTANT_InterfaceMethodref: \
                 case CONSTANT_InvokeDynamic: case CONSTANT_NameAndType
#define B1_CASES case CONSTANT_Integer: case CONSTANT_Float
#define B2_CASES case CONSTANT_Long: case CONSTANT_Double

namespace jvm_writer {
    using u1 = uint8_t;
    using u2 = uint16_t;
    using u4 = uint32_t;

    constexpr u4 MAGIC_HEADER = 0xCAFEBABE;

    template <typename T>
    inline void write_be(T int_val, u1 *&dst) {
        auto val = reinterpret_cast<std::make_unsigned_t<T>>(int_val);
        for (int i = (sizeof(T) - 1) * 8; i >= 0; i -= 8) {
            *dst++ = static_cast<u1>((val >> i) & 0xffu);
        }
    }

    template <typename T>
    inline T read_be(u1 *&src) {
        T ret = 0;
        for (auto i = 0; i < sizeof(T); i++) {
            ret <<= 8;
            ret |= *src++;
        }

        return ret;
    }

    enum ConstantPoolTag : u1 {
        CONSTANT_Class = 7, CONSTANT_Fieldref = 9,
        CONSTANT_Methodref = 10, CONSTANT_InterfaceMethodref = 11,
        CONSTANT_String = 8, CONSTANT_Integer = 3, CONSTANT_Float = 4,
        CONSTANT_Long = 5, CONSTANT_Double = 6, CONSTANT_NameAndType = 12,
        CONSTANT_Utf8 = 1, CONSTANT_MethodHandle = 15, CONSTANT_MethodType = 16,
        CONSTANT_InvokeDynamic = 18
    };

    inline std::string tag_to_string(ConstantPoolTag tag) {
        switch (tag) {
            E2S_CASE(CONSTANT_Class) E2S_CASE(CONSTANT_Fieldref)
            E2S_CASE(CONSTANT_Methodref) E2S_CASE(CONSTANT_InterfaceMethodref)
            E2S_CASE(CONSTANT_String) E2S_CASE(CONSTANT_Integer)
            E2S_CASE(CONSTANT_Float) E2S_CASE(CONSTANT_Long)
            E2S_CASE(CONSTANT_Double) E2S_CASE(CONSTANT_NameAndType)
            E2S_CASE(CONSTANT_Utf8) E2S_CASE(CONSTANT_MethodHandle)
            E2S_CASE(CONSTANT_MethodType) E2S_CASE(CONSTANT_InvokeDynamic)
            default: return "unknown tag";
        }
    }

    class cp_info {
    public:
        u1 tag;

        union alignas(size_t) {
            struct alignas(size_t) {
                u4 bytes1;
                u4 bytes2;
            };

            struct {
                u2 ref1;
                u2 ref2;
            };

            struct {
                u1 tag_kind;
                u2 method_ref;
            };

            std::string utf8;
        };

        cp_info() {};

        cp_info(cp_info &&rhs) {
            memcpy((char*) this, (char*) &rhs, sizeof(cp_info));
            if (tag == CONSTANT_Utf8)
                new (&utf8) std::string(std::move(rhs.utf8));
        }

        ~cp_info() {
            if (tag == CONSTANT_Utf8)
                utf8.~basic_string(); // this is so cursed
        }

        std::string to_string() {
            std::string ret = tag_to_string((ConstantPoolTag) tag);
            if (tag == CONSTANT_Utf8)
                ret += ": " + utf8;
            return ret;
        }

        void to_bytes(u1 *&) {

        }
        bool from_bytes(u1 *&src) {
            this->~cp_info();
            tag = *src++;
            switch (tag) {
            R1_CASES:
                ref1 = read_be<u2>(src);
                break;
            R2_CASES:
                ref1 = read_be<u2>(src);
                ref2 = read_be<u2>(src);
                break;
            B1_CASES:
                bytes1 = read_be<u4>(src);
                break;
            B2_CASES:
                bytes1 = read_be<u4>(src);
                bytes2 = read_be<u4>(src);
                break;
            case CONSTANT_MethodHandle:
                tag_kind = *src++;
                method_ref = read_be<u2>(src);
                break;
            case CONSTANT_Utf8: {
                u2 size = read_be<u2>(src);
                READER_ASSERT(new(&utf8) std::string((char *) src, (size_t) size) == &utf8,
                              "initialize string wrong addr");
                src += size;
                break;
            }
            default:
                READER_ASSERT(false, "unknown tag: " + std::to_string((int) tag));
            }
            return true;
        }
    };

    class ClassFile {
    public:
        uint16_t minor = 0x0000;
        uint16_t major = 0x0040;
        std::vector<cp_info> constants;

        void to_bytes(uint8_t *&dst) {
            write_be<uint32_t>(MAGIC_HEADER, dst);

            // minor and major versions
            write_be<uint16_t>(minor, dst);
            write_be<uint16_t>(major, dst);
        }

        bool from_bytes(uint8_t *&src) {
            READER_ASSERT(read_be<uint32_t>(src) == MAGIC_HEADER, "bad magic")
            minor = read_be<uint16_t>(src);
            major = read_be<uint16_t>(src);

            constants.clear();
            auto num_constants = read_be<uint16_t>(src) - 1;
            std::cout << "num constants: " << num_constants << '\n';
            constants.reserve(num_constants);
            for (int i = 0; i < num_constants; i++) {
                auto &back = constants.emplace_back();
                if (!back.from_bytes(src))
                    return false;
                std::cout << back.to_string() << '\n';
            }
            return true;
        }
    };
}

using namespace jvm_writer;
int main() {
    ClassFile classfile{};
    uint8_t buf[4096];
    uint8_t *ptr = buf;
    classfile.to_bytes(ptr);

    std::ios_base::fmtflags flags(std::cout.flags());
    for (int i = 0; i < 16; i++)
        std::cout << std::hex << (int) buf[i] << ' ';
    std::cout.flags(flags);
    std::cout << '\n';

    std::ifstream file("Main.class", std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (file.read(buffer.data(), size)) {
        u1 *p = (u1 *) buffer.data();
        std::cout << "buffer size: " << buffer.size() << ", ptr = " << (uint64_t) p << '\n';
        classfile.from_bytes(p);
        std::cout << classfile.major << ", " << classfile.minor << '\n';
    } else {
        std::cout << "file read failed\n";
        return 1;
    }
}