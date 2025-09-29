import React, { useState, useRef, useEffect } from 'react';
import { Send, Upload, File, Trash2, AlertCircle, CheckCircle2, Loader2 } from 'lucide-react';

const RAGChatInterface = () => {
    const [messages, setMessages] = useState([]);
    const [inputMessage, setInputMessage] = useState('');
    const [documents, setDocuments] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [selectedFile, setSelectedFile] = useState(null);
    const fileInputRef = useRef(null);
    const messagesEndRef = useRef(null);

    const API_BASE_URL = 'http://localhost:8000';

    // Auto-scroll to bottom of messages
    const scrollToBottom = () => {
        messagesEndRef.current ? .scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    // Load documents on component mount
    useEffect(() => {
        loadDocuments();
    }, []);

    const loadDocuments = async() => {
        try {
            const response = await fetch(`${API_BASE_URL}/documents`);
            if (response.ok) {
                const data = await response.json();
                setDocuments(data.documents);
            }
        } catch (error) {
            console.error('Error loading documents:', error);
        }
    };

    const handleFileSelect = (event) => {
        const file = event.target.files[0];
        if (file) {
            setSelectedFile(file);
        }
    };

    const uploadDocument = async() => {
        if (!selectedFile) return;

        setIsUploading(true);
        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch(`${API_BASE_URL}/upload-document`, {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                const result = await response.json();
                setMessages(prev => [...prev, {
                    type: 'system',
                    content: `Successfully uploaded "${result.filename}" with ${result.chunks_processed} chunks processed.`,
                    timestamp: new Date()
                }]);

                // Reload documents list
                await loadDocuments();
                setSelectedFile(null);
                fileInputRef.current.value = '';
            } else {
                const error = await response.json();
                throw new Error(error.detail);
            }
        } catch (error) {
            setMessages(prev => [...prev, {
                type: 'error',
                content: `Error uploading document: ${error.message}`,
                timestamp: new Date()
            }]);
        } finally {
            setIsUploading(false);
        }
    };

    const deleteDocument = async(docId) => {
        try {
            const response = await fetch(`${API_BASE_URL}/documents/${docId}`, {
                method: 'DELETE',
            });

            if (response.ok) {
                setMessages(prev => [...prev, {
                    type: 'system',
                    content: 'Document deleted successfully.',
                    timestamp: new Date()
                }]);
                await loadDocuments();
            } else {
                const error = await response.json();
                throw new Error(error.detail);
            }
        } catch (error) {
            setMessages(prev => [...prev, {
                type: 'error',
                content: `Error deleting document: ${error.message}`,
                timestamp: new Date()
            }]);
        }
    };

    const sendMessage = async() => {
        if (!inputMessage.trim() || isLoading) return;

        const userMessage = {
            type: 'user',
            content: inputMessage,
            timestamp: new Date()
        };

        setMessages(prev => [...prev, userMessage]);
        setInputMessage('');
        setIsLoading(true);

        try {
            // Build chat history for context
            const chatHistory = messages
                .filter(msg => msg.type === 'user' || msg.type === 'assistant')
                .slice(-6) // Last 6 messages for context
                .map(msg => ({
                    role: msg.type === 'user' ? 'user' : 'assistant',
                    content: msg.content
                }));

            const response = await fetch(`${API_BASE_URL}/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: inputMessage,
                    chat_history: chatHistory
                }),
            });

            if (response.ok) {
                const result = await response.json();

                const assistantMessage = {
                    type: 'assistant',
                    content: result.answer,
                    sources: result.sources,
                    confidence: result.confidence,
                    timestamp: new Date()
                };

                setMessages(prev => [...prev, assistantMessage]);
            } else {
                const error = await response.json();
                throw new Error(error.detail);
            }
        } catch (error) {
            setMessages(prev => [...prev, {
                type: 'error',
                content: `Error: ${error.message}`,
                timestamp: new Date()
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    const MessageComponent = ({ message }) => {
        const getMessageStyle = () => {
            switch (message.type) {
                case 'user':
                    return 'bg-blue-500 text-white ml-auto';
                case 'assistant':
                    return 'bg-gray-100 text-gray-800';
                case 'system':
                    return 'bg-green-50 text-green-800 border border-green-200';
                case 'error':
                    return 'bg-red-50 text-red-800 border border-red-200';
                default:
                    return 'bg-gray-100 text-gray-800';
            }
        };

        const getIcon = () => {
            switch (message.type) {
                case 'system':
                    return <CheckCircle2 className = "w-4 h-4" / > ;
                case 'error':
                    return <AlertCircle className = "w-4 h-4" / > ;
                default:
                    return null;
            }
        };

        return ( <
            div className = { `max-w-3xl rounded-lg p-4 mb-4 ${getMessageStyle()} ${message.type === 'user' ? 'max-w-md' : ''}` } >
            <
            div className = "flex items-start gap-2" > { getIcon() } <
            div className = "flex-1" >
            <
            div className = "whitespace-pre-wrap" > { message.content } < /div> {
                message.sources && message.sources.length > 0 && ( <
                    div className = "mt-3 pt-3 border-t border-gray-200" >
                    <
                    div className = "text-sm font-medium text-gray-600 mb-2" > Sources: < /div> <
                    div className = "space-y-1" > {
                        message.sources.map((source, index) => ( <
                            div key = { index }
                            className = "text-xs text-gray-500 flex items-center gap-2" >
                            <
                            File className = "w-3 h-3" / >
                            <
                            span > { source.source }(chunk { source.chunk_id + 1 }) < /span> <
                            span className = "text-green-600" > {
                                (source.confidence * 100).toFixed(1) } % match <
                            /span> <
                            /div>
                        ))
                    } <
                    /div> {
                        message.confidence && ( <
                            div className = "mt-2 text-xs text-gray-500" >
                            Overall confidence: {
                                (message.confidence * 100).toFixed(1) } %
                            <
                            /div>
                        )
                    } <
                    /div>
                )
            } <
            /div> <
            /div> <
            div className = "text-xs opacity-70 mt-2" > { message.timestamp.toLocaleTimeString() } < /div> <
            /div>
        );
    };

    return ( <
        div className = "flex h-screen bg-gray-50" > { /* Sidebar */ } <
        div className = "w-80 bg-white border-r border-gray-200 flex flex-col" >
        <
        div className = "p-4 border-b border-gray-200" >
        <
        h2 className = "text-lg font-semibold text-gray-800" > Document RAG Chat < /h2> <
        p className = "text-sm text-gray-500 mt-1" > Upload documents and ask questions < /p> <
        /div>

        { /* Document Upload */ } <
        div className = "p-4 border-b border-gray-200" >
        <
        div className = "space-y-3" >
        <
        input ref = { fileInputRef }
        type = "file"
        accept = ".pdf,.docx,.txt"
        onChange = { handleFileSelect }
        className = "block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100" /
        >

        {
            selectedFile && ( <
                div className = "flex items-center justify-between p-2 bg-gray-50 rounded" >
                <
                span className = "text-sm text-gray-600 truncate" > { selectedFile.name } < /span> <
                button onClick = { uploadDocument }
                disabled = { isUploading }
                className = "ml-2 px-3 py-1 bg-blue-500 text-white text-sm rounded hover:bg-blue-600 disabled:opacity-50 flex items-center gap-1" >
                {
                    isUploading ? ( <
                        Loader2 className = "w-3 h-3 animate-spin" / >
                    ) : ( <
                        Upload className = "w-3 h-3" / >
                    )
                } { isUploading ? 'Uploading...' : 'Upload' } <
                /button> <
                /div>
            )
        } <
        /div> <
        /div>

        { /* Documents List */ } <
        div className = "flex-1 overflow-y-auto" >
        <
        div className = "p-4" >
        <
        h3 className = "text-sm font-medium text-gray-700 mb-3" >
        Uploaded Documents({ documents.length }) <
        /h3>

        {
            documents.length === 0 ? ( <
                p className = "text-sm text-gray-500 italic" > No documents uploaded yet < /p>
            ) : ( <
                div className = "space-y-2" > {
                    documents.map((doc) => ( <
                        div key = { doc.id }
                        className = "flex items-center justify-between p-2 bg-gray-50 rounded" >
                        <
                        div className = "flex-1 min-w-0" >
                        <
                        div className = "text-sm font-medium text-gray-800 truncate" > { doc.filename } <
                        /div> <
                        div className = "text-xs text-gray-500" > { doc.chunks_count }
                        chunks <
                        /div> <
                        /div> <
                        button onClick = {
                            () => deleteDocument(doc.id) }
                        className = "ml-2 p-1 text-red-500 hover:text-red-700"
                        title = "Delete document" >
                        <
                        Trash2 className = "w-4 h-4" / >
                        <
                        /button> <
                        /div>
                    ))
                } <
                /div>
            )
        } <
        /div> <
        /div> <
        /div>

        { /* Main Chat Area */ } <
        div className = "flex-1 flex flex-col" > { /* Messages */ } <
        div className = "flex-1 overflow-y-auto p-4" > {
            messages.length === 0 ? ( <
                div className = "flex items-center justify-center h-full" >
                <
                div className = "text-center text-gray-500" >
                <
                File className = "w-12 h-12 mx-auto mb-4 opacity-50" / >
                <
                p className = "text-lg font-medium" > Welcome to Document RAG Chat < /p> <
                p className = "text-sm mt-2" > Upload documents and start asking questions! < /p> <
                /div> <
                /div>
            ) : ( <
                div className = "space-y-4" > {
                    messages.map((message, index) => ( <
                        MessageComponent key = { index }
                        message = { message }
                        />
                    ))
                }

                {
                    isLoading && ( <
                        div className = "flex items-center gap-2 text-gray-500" >
                        <
                        Loader2 className = "w-4 h-4 animate-spin" / >
                        <
                        span > Thinking... < /span> < /
                        div >
                    )
                }

                <
                div ref = { messagesEndRef }
                /> < /
                div >
            )
        } <
        /div>

        { /* Input Area */ } <
        div className = "border-t border-gray-200 p-4" >
        <
        div className = "flex gap-2" >
        <
        textarea value = { inputMessage }
        onChange = {
            (e) => setInputMessage(e.target.value) }
        onKeyPress = { handleKeyPress }
        placeholder = { documents.length === 0 ? "Upload documents first to start chatting..." : "Ask a question about your documents..." }
        disabled = { documents.length === 0 || isLoading }
        className = "flex-1 resize-none border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
        rows = "3" /
        >
        <
        button onClick = { sendMessage }
        disabled = {!inputMessage.trim() || documents.length === 0 || isLoading }
        className = "px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center" >
        <
        Send className = "w-4 h-4" / >
        <
        /button> <
        /div>

        <
        div className = "mt-2 text-xs text-gray-500" >
        Press Enter to send, Shift + Enter
        for new line <
        /div> <
        /div> <
        /div> <
        /div>
    );
};

export default RAGChatInterface;