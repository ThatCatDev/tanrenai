package tools

import "encoding/json"

// Schema is a JSON Schema object for describing tool parameters.
type Schema struct {
	Type       string                    `json:"type"`
	Properties map[string]SchemaProperty `json:"properties,omitempty"`
	Required   []string                  `json:"required,omitempty"`
}

// SchemaProperty describes a single property within a JSON Schema.
type SchemaProperty struct {
	Type        string `json:"type"`
	Description string `json:"description"`
}

// MustMarshal marshals the schema to json.RawMessage, panicking on error.
func (s Schema) MustMarshal() json.RawMessage {
	b, err := json.Marshal(s)
	if err != nil {
		panic("tools: failed to marshal schema: " + err.Error())
	}
	return b
}
