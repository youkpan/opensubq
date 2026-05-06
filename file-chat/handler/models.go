package handler

import (
	"encoding/json"
	"net/http"

	apimodel "file-chat/model"
)

func ModelsHandler(cfg AppConfig) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "OPTIONS" {
			w.WriteHeader(200)
			return
		}

		resp := apimodel.ModelsResponse{
			Object: "list",
			Data: []apimodel.ModelEntry{
				{
					ID:      "deepseek-v4-flash",
					Object:  "model",
					OwnedBy: "file-chat",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}
}
