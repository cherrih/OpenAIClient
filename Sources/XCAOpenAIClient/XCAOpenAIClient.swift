import Foundation
import OpenAPIRuntime
import OpenAPIURLSession

public struct OpenAIClient {
    
    public let client: Client
    private let urlSession = URLSession.shared
    private let apiKey: String
    
    public init(apiKey: String) {
        self.client = Client(
            serverURL: try! Servers.server1(),
            transport: URLSessionTransport(),
            middlewares: [AuthMiddleware(apiKey: apiKey)])
        self.apiKey = apiKey
    }
    
    /// Sends a chat completion using the model name exactly as given.
    ///
    /// The model field is `anyOf [string, enum]`, so the raw string is passed through unchanged.
    /// Previously any name missing from the (2024-era) enum was silently replaced with gpt-4.1-mini,
    /// which made newer models impossible to select without regenerating the client.
    public func promptChatGPT(
        with model: String,
        prompt: String,
        assistantPrompt: String = "You are a helpful assistant",
        responseFormatType: String? = nil,
        prevMessages: [Components.Schemas.ChatCompletionRequestMessage] = []
    ) async throws -> String {
        return try await promptChatGPT(
            prompt: prompt,
            modelPayload: .init(value1: model, value2: nil),
            assistantPrompt: assistantPrompt,
            responseFormatType: responseFormatType,
            prevMessages: prevMessages)
    }

    public func promptChatGPT4oMini(
        prompt: String,
        assistantPrompt: String = "You are a helpful assistant",
        responseFormatType: String? = nil, // Accept a string
        prevMessages: [Components.Schemas.ChatCompletionRequestMessage] = []
    ) async throws -> String {
        return try await promptChatGPT(
            prompt: prompt,
            model: .gpt_hyphen_4o_hyphen_mini,
            assistantPrompt: assistantPrompt,
            responseFormatType: responseFormatType,
            prevMessages: prevMessages)
    }
    
    public func promptChatGPT(
        prompt: String,
        model: Components.Schemas.CreateChatCompletionRequest.modelPayload.Value2Payload = .gpt_hyphen_4_period_1,
        assistantPrompt: String = "You are a helpful assistant",
        responseFormatType: String? = nil, // Accept a string
        prevMessages: [Components.Schemas.ChatCompletionRequestMessage] = []
    ) async throws -> String {
        return try await promptChatGPT(
            prompt: prompt,
            modelPayload: .init(value1: nil, value2: model),
            assistantPrompt: assistantPrompt,
            responseFormatType: responseFormatType,
            prevMessages: prevMessages)
    }

    private func promptChatGPT(
        prompt: String,
        modelPayload: Components.Schemas.CreateChatCompletionRequest.modelPayload,
        assistantPrompt: String,
        responseFormatType: String?,
        prevMessages: [Components.Schemas.ChatCompletionRequestMessage]
    ) async throws -> String {

        // Build the response_format object if responseFormatType is provided
        var responseFormat: Components.Schemas.CreateChatCompletionRequest.response_formatPayload? = nil
        if let formatType = responseFormatType {
            responseFormat = Components.Schemas.CreateChatCompletionRequest.response_formatPayload(_type: .init(rawValue: formatType))
        }

        // The instructions go in a real `system` message. They used to be sent as a prior
        // `assistant` turn, which the model treats as something it already said rather than
        // as instructions, so persona, language and "respond to the latest entry" rules were
        // frequently ignored.
        let requestBody = Components.Schemas.CreateChatCompletionRequest(
            messages: [.ChatCompletionRequestSystemMessage(.init(content: assistantPrompt, role: .system))]
            + prevMessages
            + [.ChatCompletionRequestUserMessage(.init(content: .case1(prompt), role: .user))],
            model: modelPayload,
            response_format: responseFormat // Use the built response format
        )

        let response = try await client.createChatCompletion(body: .json(requestBody))

        switch response {
        case .ok(let body):
            let json = try body.body.json
            guard let content = json.choices.first?.message.content else {
                throw "No Response"
            }
            return content
        case .undocumented(let statusCode, let payload):
            throw "OpenAIClientError - statuscode: \(statusCode), \(payload)"
        }
    }
    
    /// Defaults to gpt-4o-mini-tts: it is the only listed TTS model that honors `instructions`
    /// (tone, pacing, language). tts-1 silently ignored them.
    public func generateSpeechFrom(input: String,
                                   model: Components.Schemas.CreateSpeechRequest.modelPayload.Value2Payload = .gpt_hyphen_4o_hyphen_mini_hyphen_tts,
                                   voice: Components.Schemas.CreateSpeechRequest.voicePayload = .fable,
                                   format: Components.Schemas.CreateSpeechRequest.response_formatPayload = .aac,
                                   instructions: String = ""
    ) async throws -> Data {
        let response = try await client.createSpeech(body: .json(
            .init(
                model: .init(value1: nil, value2: model),
                input: input,
                voice: voice,
                instructions: instructions,
                response_format: format
            )))
        
        switch response {
        case .ok(let response):
            switch response.body {
            case .any(let body):
                var data = Data()
                for try await byte in body {
                    data.append(contentsOf: byte)
                }
                return data
            }
            
        case .undocumented(let statusCode, let payload):
            throw "OpenAIClientError - statuscode: \(statusCode), \(payload)"
        }
    }

    /// Use URLSession manually until swift-openapi-runtime support MultipartForm
    /// Transcribes an audio file.
    /// - Parameters:
    ///   - model: transcription model name, passed through as-is (e.g. "gpt-4o-transcribe", "gpt-transcribe").
    ///   - timeoutInterval: per-request idle timeout. Uploads of ~20-minute chunks on slow links need more than the old 30 s.
    public func generateAudioTransciptions(audioData: Data, fileName: String = "recording.m4a", prompt: String = "", languageCode: String? = nil, model: String = "gpt-4o-transcribe", timeoutInterval: TimeInterval = 120) async throws -> String {
        var request = URLRequest(url: URL(string: "https://api.openai.com/v1/audio/transcriptions")!)
        let boundary: String = UUID().uuidString
        request.timeoutInterval = timeoutInterval
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var entries: [MultipartFormDataEntry] = [
            .file(paramName: "file", fileName: fileName, fileData: audioData, contentType: "audio/mpeg"),
            .string(paramName: "model", value: model),
            .string(paramName: "response_format", value: "text"),
            .string(paramName: "prompt", value: prompt)
        ]
        if let languageCode = languageCode, !languageCode.isEmpty {
            entries.append(.string(paramName: "language", value: languageCode))
        }

        let bodyBuilder = MultipartFormDataBodyBuilder(boundary: boundary, entries: entries)

        request.httpBody = bodyBuilder.build()
        let (data, resp) = try await urlSession.data(for: request)
        guard let httpResp = resp as? HTTPURLResponse, httpResp.statusCode == 200 else {
            throw "Invalid Status Code \((resp as? HTTPURLResponse)?.statusCode ?? -1)"
        }
        guard let text = String(data: data, encoding: .utf8) else {
            throw "Invalid format"
        }
        
        return text
    }
    
    public func extractTextFromImage(base64Image: String, prompt: String = "Please extract and return all visible text content from this image, without adding anything extra.") async throws -> String {
        var request = URLRequest(url: URL(string: "https://api.openai.com/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let requestBody: [String: Any] = [
            "model": "gpt-4.1",
            "messages": [
                ["role": "user", "content": [
                    ["type": "text", "text": prompt],
                    ["type": "image_url", "image_url": [
                        "url": "data:image/jpeg;base64,\(base64Image)"
                    ]]                ]]
            ],
            "max_tokens": 2000
        ]
        
        let jsonData = try JSONSerialization.data(withJSONObject: requestBody)
        request.httpBody = jsonData
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            #if DEBUG
            print("[OpenAIClient.extractTextFromImage] ❌ Invalid HTTP status code: \((response as? HTTPURLResponse)?.statusCode ?? -1)")
            #endif
            throw "Invalid response from OpenAI Vision API"
        }
        
        struct OpenAIResponse: Decodable {
            struct Choice: Decodable {
                struct Message: Decodable {
                    let content: String
                }
                let message: Message
            }
            let choices: [Choice]
        }
        
        let decodedResponse = try JSONDecoder().decode(OpenAIResponse.self, from: data)
        
        guard let content = decodedResponse.choices.first?.message.content else {
            #if DEBUG
            print("[OpenAIClient.extractTextFromImage] ❌ No content found in response.")
            #endif
            throw "Empty response from OpenAI Vision API"
        }
        
        return content
    }
}

