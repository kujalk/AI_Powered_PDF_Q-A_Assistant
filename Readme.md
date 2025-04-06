# 📄 PDF Q&A Assistant

An interactive web application that allows users to upload PDF documents and ask questions about their content using various AI models.

![PDF Q&A Assistant](images/app-frontend.png)

## Features

- **Multiple AI Provider Support**: OpenAI, Google Gemini, and local Mistral models
- **PDF Processing**: Upload and analyze PDF documents of any size
- **Interactive Q&A**: Ask questions about your document content
- **API Usage Tracking**: Monitor token usage and estimated costs
- **Responsive UI**: Clean, modern interface built with Streamlit

## Screenshots

### Initial Interface
![Initial Interface](images/app-frontend.png)

### OpenAI Model Processing
![OpenAI Processing](images/openai-processing.png)

### Google Gemini Integration
![Google Gemini](images/gemini-processing.png)

### Local Mistral Processing
![Local Mistral](images/mistral-local.png)

## Installation

1. Clone this repository
```bash
git clone https://github.com/johndoe/pdf-qa-assistant.git
cd pdf-qa-assistant
```

2. Install the required dependencies
```bash
pip install -r requirements.txt
```

3. Run the application
```bash
streamlit run app_gemini.py
```

## Requirements

```
streamlit
PyPDF2
langchain
langchain-community
faiss-cpu
openai
google-generativeai
langchain-google-genai
requests
```

## API Setup

### OpenAI
1. Create an account at [OpenAI](https://platform.openai.com/signup)
2. Generate an API key at [API Keys](https://platform.openai.com/account/api-keys)
3. Paste the key in the application sidebar

### Google Gemini
1. Create an account at [Google AI Studio](https://ai.google.dev/)
2. Get your API key from [API Keys](https://console.cloud.google.com/apis/credentials)
3. Paste the key in the application sidebar

### Local Mistral
1. Install Ollama: [ollama.com](https://ollama.com)
2. Pull the model:
```bash
ollama pull mistral:7b-instruct-q4_0
```
3. Run the server:
```bash
ollama serve
```

4. To check which models your Gemini can support
```bash
curl "https://generativelanguage.googleapis.com/v1beta/models?key=YOUR_GEMINI_API_KEY"
```

## How It Works

1. **Upload a PDF**: The application extracts text from your PDF document
2. **Process with AI**: The text is chunked and embedded using the selected AI model
3. **Ask Questions**: Type questions about your document to get AI-powered answers
4. **Track Usage**: Monitor API calls, token usage, and costs in real-time

## Technical Architecture

- **Frontend**: Streamlit for UI components
- **Text Processing**: PyPDF2 for PDF text extraction
- **Embedding**: Model-specific embedding services (OpenAI, Google, or Ollama)
- **Vector Storage**: FAISS for efficient similarity search
- **LLM Integration**: LangChain for connecting to various AI models

## Use Cases

- **Research**: Quickly extract insights from academic papers or research documents
- **Legal Document Analysis**: Search through complex legal documents for specific clauses
- **Knowledge Management**: Extract information from technical documentation
- **Education**: Analyze and question educational materials

## About the Developer

PDF Q&A Assistant is developed by [John Doe](https://github.com/johndoe).

For more insights on AI and development, check out my [WordPress blog](https://johndoe.wordpress.com).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request