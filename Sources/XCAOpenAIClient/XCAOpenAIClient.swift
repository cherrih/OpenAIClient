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
    
    public func promptChatGPT(
        with model: String,
        prompt: String,
        assistantPrompt: String = "You are a helpful assistant",
        responseFormatType: String? = nil, 
        prevMessages: [Components.Schemas.ChatCompletionRequestMessage] = []
    ) async throws -> String {
        
        var modelPayload: Components.Schemas.CreateChatCompletionRequest.modelPayload.Value2Payload = .gpt_hyphen_4_period_1_hyphen_mini
        if let value = Components.Schemas.CreateChatCompletionRequest.modelPayload.Value2Payload(rawValue: model) {
            modelPayload = value
        }
        
        print("RUNNING WITH MODEL \(modelPayload)")

        return try await promptChatGPT(
            prompt: prompt,
            model: modelPayload,
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

        // Build the response_format object if responseFormatType is provided
        var responseFormat: Components.Schemas.CreateChatCompletionRequest.response_formatPayload? = nil
        if let formatType = responseFormatType {
            responseFormat = Components.Schemas.CreateChatCompletionRequest.response_formatPayload(_type: .init(rawValue: formatType))
        }

        // Build the body for the request
        let requestBody = Components.Schemas.CreateChatCompletionRequest(
            messages: [.ChatCompletionRequestAssistantMessage(.init(content: assistantPrompt, role: .assistant))]
            + prevMessages
            + [.ChatCompletionRequestUserMessage(.init(content: .case1(prompt), role: .user))],
            model: .init(value1: nil, value2: model),
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
    
    public func generateSpeechFrom(input: String,
                                   model: Components.Schemas.CreateSpeechRequest.modelPayload.Value2Payload = .tts_hyphen_1,
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
    public func generateAudioTransciptions(audioData: Data, fileName: String = "recording.m4a", prompt: String = "", languageCode: String? = nil) async throws -> String {
        var request = URLRequest(url: URL(string: "https://api.openai.com/v1/audio/transcriptions")!)
        let boundary: String = UUID().uuidString
        request.timeoutInterval = 30
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var entries: [MultipartFormDataEntry] = [
            .file(paramName: "file", fileName: fileName, fileData: audioData, contentType: "audio/mpeg"),
            .string(paramName: "model", value: "gpt-4o-transcribe"),
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
    
}
