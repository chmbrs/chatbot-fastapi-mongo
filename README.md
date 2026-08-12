# chatbot-fastapi-mongo

## Quickstart (No API Key Required)
You can run the app immediately using the built-in offline demo provider. 

1. Start the container:
```bash
docker compose up
```
2. Open the app at http://localhost:8501/. 

## Using Real Models
By default, the app uses a demo provider. You can configure it to use real cloud or local models:

* **Cloud Model (OpenRouter):**
  * Get an API key from OpenRouter (Note: the free tier is limited to 20 requests/minute and 50/day).
  * Copy `.env.example` to `.env` and set `LLM_API_KEY=sk-or-...`
  * Run `docker compose up`.

* **Local Model (Ollama):**
  * Run `ollama pull gemma4:latest`.
  * **Via UI:** If no API key is set, simply flip the "Use local Ollama model" toggle in the sidebar.
  * **Via Config:** To make it the default, set `LLM_PROVIDER=ollama` in your `.env` file and rebuild.

## Key Features & Demo Constraints
The standout feature is **conversations that message each other**. 

Watch the demo: https://github.com/user-attachments/assets/0bbdfaef-54fc-4557-bbda-bdd190bc2659

However, because this is designed as a playable demo rather than a production-ready messaging queue, it deliberately excludes certain complexities:

* **No Message Expiry (TTL):** Held messages wait indefinitely for user action.
* **No Cross-Machine Delivery:** All sessions and conversations exist strictly within the same database.
* **No Unsupervised Loops:** There are no queue caps or rate limits because automatic exchanges *strictly alternate* between the two starting conversations. 
* **No Streaming on `/send`:** The generated output lands in the receiving conversation, meaning there is no response for the sender to stream.
