package database

import (
	"testing"
)

func TestEncryptDecryptRoundTrip(t *testing.T) {
	key, err := GenerateEncryptionKey()
	if err != nil {
		t.Fatalf("generate key: %v", err)
	}

	plaintext := []byte("sk-vast-ai-test-key-12345")

	ciphertext, err := Encrypt(plaintext, key)
	if err != nil {
		t.Fatalf("encrypt: %v", err)
	}

	if string(ciphertext) == string(plaintext) {
		t.Fatal("ciphertext should differ from plaintext")
	}

	decrypted, err := Decrypt(ciphertext, key)
	if err != nil {
		t.Fatalf("decrypt: %v", err)
	}

	if string(decrypted) != string(plaintext) {
		t.Fatalf("decrypted = %q, want %q", string(decrypted), string(plaintext))
	}
}

func TestDecryptWrongKey(t *testing.T) {
	key1, _ := GenerateEncryptionKey()
	key2, _ := GenerateEncryptionKey()

	ciphertext, _ := Encrypt([]byte("secret"), key1)

	_, err := Decrypt(ciphertext, key2)
	if err == nil {
		t.Fatal("decrypt with wrong key should fail")
	}
}

func TestEncryptBadKey(t *testing.T) {
	_, err := Encrypt([]byte("test"), "too-short")
	if err == nil {
		t.Fatal("encrypt with bad key should fail")
	}
}

func TestGenerateEncryptionKey(t *testing.T) {
	key, err := GenerateEncryptionKey()
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if len(key) != 64 { // 32 bytes = 64 hex chars
		t.Fatalf("key length = %d, want 64 hex chars", len(key))
	}
}
