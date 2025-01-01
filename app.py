from flask import Flask, render_template_string, request, jsonify
from transformers import LEDForConditionalGeneration, LEDTokenizer
import torch
import time

app = Flask(__name__)

# HTML template as a string
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Legal Document Summarizer</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/marked@4.0.0/marked.min.js"></script>
    <style>
        .gradient-text {
            background: linear-gradient(45deg, #4F46E5, #7C3AED);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .loading {
            display: inline-block;
            width: 30px;
            height: 30px;
            border: 3px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s ease-in-out infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <!-- Header -->
        <div class="text-center mb-12">
            <h1 class="text-4xl font-bold mb-4 gradient-text">Legal Document Summarizer</h1>
            <p class="text-gray-600">Indian Legal Documents AI Summarization</p>
        </div>

        <!-- Main Content -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
            <!-- Input Section -->
            <div class="bg-white rounded-lg shadow-lg p-6">
                <div class="flex justify-between items-center mb-4">
                    <h2 class="text-xl font-semibold text-gray-800">Input Document</h2>
                    <span id="wordCount" class="text-sm text-gray-500">0 words</span>
                </div>
                <textarea
                    id="inputText"
                    class="w-full h-[500px] p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                    placeholder="Paste your legal document here..."></textarea>
                <div class="mt-4 flex justify-end">
                    <button
                        id="summarizeBtn"
                        class="bg-indigo-600 text-white px-6 py-2 rounded-lg hover:bg-indigo-700 transition-colors flex items-center gap-2">
                        <span>Summarize</span>
                        <div id="loading" class="loading hidden"></div>
                    </button>
                </div>
            </div>

            <!-- Output Section -->
            <div class="bg-white rounded-lg shadow-lg p-6">
                <div class="flex justify-between items-center mb-4">
                    <h2 class="text-xl font-semibold text-gray-800">Summary</h2>
                    <button
                        id="copyBtn"
                        class="text-gray-600 px-4 py-1 rounded-lg hover:bg-gray-100 transition-colors hidden">
                        Copy
                    </button>
                </div>
                <div id="summary" class="h-[500px] overflow-auto p-4 border border-gray-300 rounded-lg bg-gray-50 prose">
                    <p class="text-gray-500 italic">Summary will appear here...</p>
                </div>
                <div class="mt-4 text-right">
                    <span id="processingTime" class="text-sm text-gray-500"></span>
                </div>
            </div>
        </div>

        <!-- Sample Documents -->
        <div class="mt-12">
            <h3 class="text-xl font-semibold mb-4 text-gray-800">Sample Documents</h3>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <button onclick="loadSample(0)" class="bg-white p-4 rounded-lg shadow hover:shadow-md transition-shadow text-left">
                    <h4 class="font-medium text-gray-800">Property Rights Case</h4>
                    <p class="text-sm text-gray-600 mt-1">Supreme Court judgment on property rights...</p>
                </button>
                <button onclick="loadSample(1)" class="bg-white p-4 rounded-lg shadow hover:shadow-md transition-shadow text-left">
                    <h4 class="font-medium text-gray-800">Constitutional Matter</h4>
                    <p class="text-sm text-gray-600 mt-1">Constitutional validity challenge...</p>
                </button>
                <button onclick="loadSample(2)" class="bg-white p-4 rounded-lg shadow hover:shadow-md transition-shadow text-left">
                    <h4 class="font-medium text-gray-800">Criminal Appeal</h4>
                    <p class="text-sm text-gray-600 mt-1">Criminal appeal regarding evidence...</p>
                </button>
            </div>
        </div>
    </div>

    <script>
        const sampleTexts = [
            `IN THE SUPREME COURT OF INDIA
CIVIL APPELLATE JURISDICTION
Civil Appeal No. 123456 of 2023

ABC Developers                           ...Appellant
Versus
State of Maharashtra & Ors.              ...Respondents

JUDGMENT
The present appeal challenges the judgment dated 15.03.2023 passed by the High Court of Bombay, which upheld the order of the revenue authorities cancelling certain land allotments. The core issue relates to interpretation of development agreements and property rights under Maharashtra Land Revenue Code...`,

            `IN THE HIGH COURT OF DELHI
WRIT PETITION (CIVIL) NO. 789 OF 2023

IN THE MATTER OF:
Challenge to Constitutional Validity of Section 6A of Delhi Rent Control Act

The petitioner has challenged the constitutional validity of the recent amendment to the Delhi Rent Control Act, specifically Section 6A, on grounds of violation of Article 14 and Article 19(1)(g) of the Constitution...`,

            `IN THE SUPREME COURT OF INDIA
CRIMINAL APPELLATE JURISDICTION
Criminal Appeal No. 567 of 2023

State of Karnataka                       ...Appellant
Versus
Mr. XYZ                                 ...Respondent

This criminal appeal arises from the judgment of Karnataka High Court acquitting the respondent. The primary question relates to admissibility of electronic evidence and interpretation of Section 65B of Indian Evidence Act...`
        ];

        function loadSample(index) {
            document.getElementById('inputText').value = sampleTexts[index];
            updateWordCount();
        }

        function updateWordCount() {
            const text = document.getElementById('inputText').value;
            const wordCount = text.trim().split(/\s+/).length;
            document.getElementById('wordCount').textContent = `${wordCount} words`;
        }

        document.getElementById('inputText').addEventListener('input', updateWordCount);

        document.getElementById('summarizeBtn').addEventListener('click', async () => {
            const text = document.getElementById('inputText').value;
            if (!text.trim()) return;

            // Show loading state
            document.getElementById('loading').classList.remove('hidden');
            document.getElementById('summarizeBtn').disabled = true;
            document.getElementById('summary').innerHTML = '<p class="text-gray-500">Generating summary...</p>';

            try {
                const response = await fetch('/summarize', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text }),
                });

                const data = await response.json();

                // Update summary
                document.getElementById('summary').innerHTML = marked.parse(data.summary);
                document.getElementById('processingTime').textContent = `Processing time: ${data.processing_time}s`;
                document.getElementById('copyBtn').classList.remove('hidden');
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('summary').innerHTML = '<p class="text-red-500">Error generating summary. Please try again.</p>';
            } finally {
                document.getElementById('loading').classList.add('hidden');
                document.getElementById('summarizeBtn').disabled = false;
            }
        });

        document.getElementById('copyBtn').addEventListener('click', () => {
            const summaryText = document.getElementById('summary').textContent;
            navigator.clipboard.writeText(summaryText);

            const btn = document.getElementById('copyBtn');
            btn.textContent = 'Copied!';
            btn.classList.add('bg-green-100');

            setTimeout(() => {
                btn.textContent = 'Copy';
                btn.classList.remove('bg-green-100');
            }, 2000);
        });
    </script>
</body>
</html>
"""

class LegalSummarizer:
    def __init__(self, model_path="/home/nishant/Desktop/AI/AI_Final/models/legal_led_final", device="cuda:0"):
        self.device = torch.device(device)
        self.model = LEDForConditionalGeneration.from_pretrained(model_path).to(self.device)
        self.tokenizer = LEDTokenizer.from_pretrained(model_path)
        self.model.eval()

    def summarize(self, text, max_length=2048, min_length=50):
        inputs = self.tokenizer(
            text,
            max_length=8000,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            start_time = time.time()
            summary_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
            proc_time = time.time() - start_time

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary, proc_time

# Initialize summarizer
summarizer = LegalSummarizer()

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.json['text']
    summary, proc_time = summarizer.summarize(text)
    return jsonify({
        'summary': summary,
        'processing_time': f"{proc_time:.2f}"
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
