package gguf

import (
	"bytes"
	"encoding/binary"
	"io"
	"testing"
)

// ggufBuilder helps construct minimal GGUF files for testing.
type ggufBuilder struct {
	buf     bytes.Buffer
	version uint32
	kvs     []kv
}

type kv struct {
	key       string
	valueType uint32
	value     interface{} // string, uint32, int32, float32, bool, []string, etc.
}

func newBuilder(version uint32) *ggufBuilder {
	return &ggufBuilder{version: version}
}

func (b *ggufBuilder) addKV(key string, valueType uint32, value interface{}) {
	b.kvs = append(b.kvs, kv{key: key, valueType: valueType, value: value})
}

func (b *ggufBuilder) build() *bytes.Reader {
	b.buf.Reset()

	// Magic
	binary.Write(&b.buf, binary.LittleEndian, uint32(ggufMagic))
	// Version
	binary.Write(&b.buf, binary.LittleEndian, b.version)
	// Tensor count, KV count
	if b.version == 2 {
		binary.Write(&b.buf, binary.LittleEndian, uint32(0)) // tensors
		binary.Write(&b.buf, binary.LittleEndian, uint32(len(b.kvs)))
	} else {
		binary.Write(&b.buf, binary.LittleEndian, uint64(0)) // tensors
		binary.Write(&b.buf, binary.LittleEndian, uint64(len(b.kvs)))
	}

	for _, kv := range b.kvs {
		b.writeString(kv.key)
		binary.Write(&b.buf, binary.LittleEndian, kv.valueType)
		b.writeValue(kv.valueType, kv.value)
	}

	return bytes.NewReader(b.buf.Bytes())
}

func (b *ggufBuilder) writeString(s string) {
	if b.version == 2 {
		binary.Write(&b.buf, binary.LittleEndian, uint32(len(s)))
	} else {
		binary.Write(&b.buf, binary.LittleEndian, uint64(len(s)))
	}
	b.buf.WriteString(s)
}

func (b *ggufBuilder) writeValue(valueType uint32, value interface{}) {
	switch valueType {
	case valueTypeString:
		b.writeString(value.(string))
	case valueTypeUint8:
		binary.Write(&b.buf, binary.LittleEndian, value.(uint8))
	case valueTypeInt8:
		binary.Write(&b.buf, binary.LittleEndian, value.(int8))
	case valueTypeUint16:
		binary.Write(&b.buf, binary.LittleEndian, value.(uint16))
	case valueTypeInt16:
		binary.Write(&b.buf, binary.LittleEndian, value.(int16))
	case valueTypeUint32:
		binary.Write(&b.buf, binary.LittleEndian, value.(uint32))
	case valueTypeInt32:
		binary.Write(&b.buf, binary.LittleEndian, value.(int32))
	case valueTypeFloat32:
		binary.Write(&b.buf, binary.LittleEndian, value.(float32))
	case valueTypeBool:
		if value.(bool) {
			b.buf.WriteByte(1)
		} else {
			b.buf.WriteByte(0)
		}
	case valueTypeUint64:
		binary.Write(&b.buf, binary.LittleEndian, value.(uint64))
	case valueTypeInt64:
		binary.Write(&b.buf, binary.LittleEndian, value.(int64))
	case valueTypeFloat64:
		binary.Write(&b.buf, binary.LittleEndian, value.(float64))
	case valueTypeArray:
		arr := value.([]string)
		binary.Write(&b.buf, binary.LittleEndian, uint32(valueTypeString)) // elem type
		if b.version == 2 {
			binary.Write(&b.buf, binary.LittleEndian, uint32(len(arr)))
		} else {
			binary.Write(&b.buf, binary.LittleEndian, uint64(len(arr)))
		}
		for _, s := range arr {
			b.writeString(s)
		}
	}
}

func TestReadMetadata_V3_AllKeys(t *testing.T) {
	b := newBuilder(3)
	b.addKV("general.architecture", valueTypeString, "qwen2")
	b.addKV("general.name", valueTypeString, "Qwen2.5-Coder-32B-Instruct")
	b.addKV("tokenizer.chat_template", valueTypeString, "{{ messages }}")

	meta, err := readMetadataFrom(b.build())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if meta.Version != 3 {
		t.Errorf("version = %d, want 3", meta.Version)
	}
	if meta.Architecture != "qwen2" {
		t.Errorf("architecture = %q, want %q", meta.Architecture, "qwen2")
	}
	if meta.Name != "Qwen2.5-Coder-32B-Instruct" {
		t.Errorf("name = %q, want %q", meta.Name, "Qwen2.5-Coder-32B-Instruct")
	}
	if meta.ChatTemplate != "{{ messages }}" {
		t.Errorf("chat_template = %q, want %q", meta.ChatTemplate, "{{ messages }}")
	}
}

func TestReadMetadata_V2(t *testing.T) {
	b := newBuilder(2)
	b.addKV("general.architecture", valueTypeString, "llama")
	b.addKV("general.name", valueTypeString, "Llama-3")

	meta, err := readMetadataFrom(b.build())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if meta.Version != 2 {
		t.Errorf("version = %d, want 2", meta.Version)
	}
	if meta.Architecture != "llama" {
		t.Errorf("architecture = %q, want %q", meta.Architecture, "llama")
	}
	if meta.Name != "Llama-3" {
		t.Errorf("name = %q, want %q", meta.Name, "Llama-3")
	}
}

func TestReadMetadata_SkipNonStringTargets(t *testing.T) {
	// If a target key has a non-string value type, it should be skipped.
	b := newBuilder(3)
	b.addKV("general.architecture", valueTypeUint32, uint32(42)) // non-string, skip
	b.addKV("general.name", valueTypeString, "TestModel")

	meta, err := readMetadataFrom(b.build())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if meta.Architecture != "" {
		t.Errorf("architecture = %q, want empty (was non-string)", meta.Architecture)
	}
	if meta.Name != "TestModel" {
		t.Errorf("name = %q, want %q", meta.Name, "TestModel")
	}
}

func TestReadMetadata_SkipAllValueTypes(t *testing.T) {
	// Ensure we can skip every fixed-size value type, plus arrays.
	b := newBuilder(3)
	b.addKV("skip.uint8", valueTypeUint8, uint8(1))
	b.addKV("skip.int8", valueTypeInt8, int8(-1))
	b.addKV("skip.uint16", valueTypeUint16, uint16(2))
	b.addKV("skip.int16", valueTypeInt16, int16(-2))
	b.addKV("skip.uint32", valueTypeUint32, uint32(3))
	b.addKV("skip.int32", valueTypeInt32, int32(-3))
	b.addKV("skip.float32", valueTypeFloat32, float32(1.5))
	b.addKV("skip.bool", valueTypeBool, true)
	b.addKV("skip.uint64", valueTypeUint64, uint64(4))
	b.addKV("skip.int64", valueTypeInt64, int64(-4))
	b.addKV("skip.float64", valueTypeFloat64, float64(2.5))
	b.addKV("skip.string", valueTypeString, "hello")
	b.addKV("skip.array", valueTypeArray, []string{"a", "b"})
	b.addKV("general.architecture", valueTypeString, "found")

	meta, err := readMetadataFrom(b.build())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if meta.Architecture != "found" {
		t.Errorf("architecture = %q, want %q", meta.Architecture, "found")
	}
}

func TestReadMetadata_InvalidMagic(t *testing.T) {
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, uint32(0xDEADBEEF))

	_, err := readMetadataFrom(bytes.NewReader(buf.Bytes()))
	if err == nil {
		t.Fatal("expected error for invalid magic")
	}
}

func TestReadMetadata_UnsupportedVersion(t *testing.T) {
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, uint32(ggufMagic))
	binary.Write(&buf, binary.LittleEndian, uint32(99))

	_, err := readMetadataFrom(bytes.NewReader(buf.Bytes()))
	if err == nil {
		t.Fatal("expected error for unsupported version")
	}
}

func TestReadMetadata_TruncatedFile(t *testing.T) {
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, uint32(ggufMagic))
	// No version — truncated

	_, err := readMetadataFrom(bytes.NewReader(buf.Bytes()))
	if err == nil {
		t.Fatal("expected error for truncated file")
	}
}

func TestReadMetadata_EmptyFile(t *testing.T) {
	_, err := readMetadataFrom(bytes.NewReader(nil))
	if err == nil {
		t.Fatal("expected error for empty file")
	}
}

func TestReadMetadata_NoTargetKeys(t *testing.T) {
	b := newBuilder(3)
	b.addKV("other.key", valueTypeString, "value")

	meta, err := readMetadataFrom(b.build())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if meta.Architecture != "" || meta.Name != "" || meta.ChatTemplate != "" {
		t.Errorf("expected all empty, got arch=%q name=%q tpl=%q",
			meta.Architecture, meta.Name, meta.ChatTemplate)
	}
}

// readMetadataFrom wraps the internal reader for testing with bytes.Reader.
// We need to make the function work with io.ReadSeeker.
func init() {
	// Verify bytes.Reader implements io.ReadSeeker.
	var _ io.ReadSeeker = (*bytes.Reader)(nil)
}
