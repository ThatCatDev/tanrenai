package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// GGUF magic number: "GGUF" in little-endian.
const ggufMagic = 0x46554747 // 'G','G','U','F'

// GGUF value types.
const (
	valueTypeUint8   = 0
	valueTypeInt8    = 1
	valueTypeUint16  = 2
	valueTypeInt16   = 3
	valueTypeUint32  = 4
	valueTypeInt32   = 5
	valueTypeFloat32 = 6
	valueTypeBool    = 7
	valueTypeString  = 8
	valueTypeArray   = 9
	valueTypeUint64  = 10
	valueTypeInt64   = 11
	valueTypeFloat64 = 12
)

// Fixed byte sizes for non-variable value types.
var valueFixedSize = map[uint32]int64{
	valueTypeUint8:   1,
	valueTypeInt8:    1,
	valueTypeUint16:  2,
	valueTypeInt16:   2,
	valueTypeUint32:  4,
	valueTypeInt32:   4,
	valueTypeFloat32: 4,
	valueTypeBool:    1,
	valueTypeUint64:  8,
	valueTypeInt64:   8,
	valueTypeFloat64: 8,
}

// Target metadata keys.
const (
	keyArchitecture = "general.architecture"
	keyName         = "general.name"
	keyChatTemplate = "tokenizer.chat_template"
)

// Metadata holds selected fields extracted from a GGUF file header.
type Metadata struct {
	Version      uint32
	Architecture string // general.architecture
	Name         string // general.name
	ChatTemplate string // tokenizer.chat_template
}

// ReadMetadata reads only the GGUF header and metadata KV pairs from the given
// file. It does not read tensor data. Supports GGUF v2 and v3.
func ReadMetadata(path string) (*Metadata, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("gguf: open: %w", err)
	}
	defer f.Close()
	return readMetadataFrom(f)
}

func readMetadataFrom(r io.ReadSeeker) (*Metadata, error) {
	// Read magic.
	var magic uint32
	if err := binary.Read(r, binary.LittleEndian, &magic); err != nil {
		return nil, fmt.Errorf("gguf: read magic: %w", err)
	}
	if magic != ggufMagic {
		return nil, fmt.Errorf("gguf: invalid magic 0x%08X (expected 0x%08X)", magic, ggufMagic)
	}

	// Read version.
	var version uint32
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("gguf: read version: %w", err)
	}
	if version < 2 || version > 3 {
		return nil, fmt.Errorf("gguf: unsupported version %d", version)
	}

	// Read tensor count and KV count.
	// v2 uses uint32, v3 uses uint64.
	var tensorCount, kvCount uint64
	if version == 2 {
		var tc, kc uint32
		if err := binary.Read(r, binary.LittleEndian, &tc); err != nil {
			return nil, fmt.Errorf("gguf: read tensor count: %w", err)
		}
		if err := binary.Read(r, binary.LittleEndian, &kc); err != nil {
			return nil, fmt.Errorf("gguf: read kv count: %w", err)
		}
		tensorCount, kvCount = uint64(tc), uint64(kc)
	} else {
		if err := binary.Read(r, binary.LittleEndian, &tensorCount); err != nil {
			return nil, fmt.Errorf("gguf: read tensor count: %w", err)
		}
		if err := binary.Read(r, binary.LittleEndian, &kvCount); err != nil {
			return nil, fmt.Errorf("gguf: read kv count: %w", err)
		}
	}
	_ = tensorCount // we only need KV pairs

	meta := &Metadata{Version: version}
	found := 0

	for i := uint64(0); i < kvCount; i++ {
		key, err := readString(r, version)
		if err != nil {
			return nil, fmt.Errorf("gguf: kv %d key: %w", i, err)
		}

		var valueType uint32
		if err := binary.Read(r, binary.LittleEndian, &valueType); err != nil {
			return nil, fmt.Errorf("gguf: kv %d value type: %w", i, err)
		}

		// Check if this is a key we want.
		isTarget := key == keyArchitecture || key == keyName || key == keyChatTemplate
		if isTarget && valueType == valueTypeString {
			val, err := readString(r, version)
			if err != nil {
				return nil, fmt.Errorf("gguf: kv %d value: %w", i, err)
			}
			switch key {
			case keyArchitecture:
				meta.Architecture = val
			case keyName:
				meta.Name = val
			case keyChatTemplate:
				meta.ChatTemplate = val
			}
			found++
			if found == 3 {
				break // got all target keys
			}
		} else {
			// Skip this value.
			if err := skipValue(r, version, valueType); err != nil {
				return nil, fmt.Errorf("gguf: kv %d skip: %w", i, err)
			}
		}
	}

	return meta, nil
}

// readString reads a GGUF string (length-prefixed).
// v2 uses uint32 lengths, v3 uses uint64.
func readString(r io.Reader, version uint32) (string, error) {
	var length uint64
	if version == 2 {
		var l uint32
		if err := binary.Read(r, binary.LittleEndian, &l); err != nil {
			return "", err
		}
		length = uint64(l)
	} else {
		if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
			return "", err
		}
	}

	if length > 10*1024*1024 { // 10 MB sanity limit
		return "", fmt.Errorf("string length %d exceeds sanity limit", length)
	}

	buf := make([]byte, length)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

// skipValue skips a value of the given type in the reader.
func skipValue(r io.ReadSeeker, version, valueType uint32) error {
	if sz, ok := valueFixedSize[valueType]; ok {
		_, err := r.Seek(sz, io.SeekCurrent)
		return err
	}

	switch valueType {
	case valueTypeString:
		_, err := readString(r, version)
		return err

	case valueTypeArray:
		// Array header: element type (uint32) + count (uint64 for v3, uint32 for v2).
		var elemType uint32
		if err := binary.Read(r, binary.LittleEndian, &elemType); err != nil {
			return err
		}
		var count uint64
		if version == 2 {
			var c uint32
			if err := binary.Read(r, binary.LittleEndian, &c); err != nil {
				return err
			}
			count = uint64(c)
		} else {
			if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
				return err
			}
		}

		// If elements have fixed size, skip in one seek.
		if sz, ok := valueFixedSize[elemType]; ok {
			_, err := r.Seek(int64(count)*sz, io.SeekCurrent)
			return err
		}

		// Otherwise skip each element.
		for j := uint64(0); j < count; j++ {
			if err := skipValue(r, version, elemType); err != nil {
				return err
			}
		}
		return nil

	default:
		return fmt.Errorf("unknown value type %d", valueType)
	}
}
