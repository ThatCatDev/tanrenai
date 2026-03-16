package tools

import (
	"encoding/json"
	"testing"
)

func TestMustMarshal(t *testing.T) {
	s := Schema{
		Type: "object",
		Properties: map[string]SchemaProperty{
			"path":    {Type: "string", Description: "Path to file"},
			"content": {Type: "string", Description: "File content"},
		},
		Required: []string{"path", "content"},
	}

	raw := s.MustMarshal()
	if raw == nil {
		t.Fatal("expected non-nil json.RawMessage")
	}
	if len(raw) == 0 {
		t.Fatal("expected non-empty json.RawMessage")
	}

	// Verify it is valid JSON.
	var roundTrip map[string]interface{}
	if err := json.Unmarshal(raw, &roundTrip); err != nil {
		t.Fatalf("MustMarshal produced invalid JSON: %v", err)
	}

	// Verify type field.
	if got, ok := roundTrip["type"]; !ok || got != "object" {
		t.Errorf("expected type='object' in marshalled schema, got: %v", got)
	}

	// Verify properties are present.
	props, ok := roundTrip["properties"].(map[string]interface{})
	if !ok {
		t.Fatal("expected 'properties' field in marshalled schema")
	}
	if _, ok := props["path"]; !ok {
		t.Error("expected 'path' property in schema")
	}
	if _, ok := props["content"]; !ok {
		t.Error("expected 'content' property in schema")
	}

	// Verify required array.
	req, ok := roundTrip["required"].([]interface{})
	if !ok {
		t.Fatal("expected 'required' array in marshalled schema")
	}
	if len(req) != 2 {
		t.Errorf("expected 2 required fields, got %d", len(req))
	}
}

func TestMustMarshalEmptySchema(t *testing.T) {
	s := Schema{Type: "object"}
	raw := s.MustMarshal()
	if raw == nil || len(raw) == 0 {
		t.Fatal("expected non-empty output for minimal schema")
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		t.Fatalf("MustMarshal produced invalid JSON for empty schema: %v", err)
	}
}

func TestSchemaPropertyFields(t *testing.T) {
	sp := SchemaProperty{Type: "integer", Description: "a number"}
	b, err := json.Marshal(sp)
	if err != nil {
		t.Fatalf("failed to marshal SchemaProperty: %v", err)
	}
	var m map[string]interface{}
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("invalid JSON from SchemaProperty: %v", err)
	}
	if m["type"] != "integer" {
		t.Errorf("expected type='integer', got: %v", m["type"])
	}
	if m["description"] != "a number" {
		t.Errorf("expected description='a number', got: %v", m["description"])
	}
}
