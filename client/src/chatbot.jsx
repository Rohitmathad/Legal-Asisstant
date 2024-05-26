// import React, { useEffect } from 'react'
//  import './chatbot.css'
// const Chatbot = () => {
//   useEffect(() => {
//     const script = document.createElement('script')
//     script.src = 'https://cdn.botpress.cloud/webchat/v1/inject.js'
//     script.async = true
//     document.body.appendChild(script)
 
//     script.onload = () => {
//       window.botpressWebChat.init({
       
//         "composerPlaceholder": "Ask your legal query",
//       "botConversationDescription": "",
//       "botId": "42cc66eb-b6fb-44f8-9d2a-2b732298f460",
//       "hostUrl": "https://cdn.botpress.cloud/webchat/v1",
//       "messagingUrl": "https://messaging.botpress.cloud",
//       "clientId": "42cc66eb-b6fb-44f8-9d2a-2b732298f460",
//       "lazySocket": true,
//       "botName": "LegalAdvisor",
//       "stylesheet": "https://webchat-styler-css.botpress.app/prod/code/a1ecdf9f-eecd-49d0-af7d-626060e2799b/v28935/style.css",
//       "frontendVersion": "v1"
        
//       })
//     }
//   }, [])
 
//   return (

//   <>
//   <div className='pb-1'>
  
//   <div id="webchat" />
//   </div>
    
//   </>
//   )
// }
 
// export default Chatbot


import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const App = () => {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [processingComplete, setProcessingComplete] = useState(false);

  useEffect(() => {
    const handleProcessPdf = async () => {
      try {
        const response = await axios.post('http://127.0.0.1:5000/api/process_pdf');
        alert(response.data.message);
        setProcessingComplete(true);
      } catch (error) {
        console.error("There was an error processing the PDF!", error);
      }
    };

    handleProcessPdf();
  }, []);

  const handleQuestionSubmit = async () => {
    if (!question) {
      alert("Please enter a question.");
      return;
    }

    try {
      const response = await axios.post('http://localhost:5000/ask_question', { question });
      setAnswer(response.data.answer);
    } catch (error) {
      console.error("There was an error asking the question!", error);
    }
  };

  return (
    <>
      <div className="App">
        <h1>Chat with PDF using Gemini💁</h1>
        {!processingComplete && <p>Processing PDF, please wait...</p>}
        {processingComplete && (
          <div>
            <input 
              type="text" 
              value={question} 
              onChange={(e) => setQuestion(e.target.value)} 
              placeholder="Ask a question from the PDF" 
            />
            <button onClick={handleQuestionSubmit}>Ask</button>
          </div>
        )}
        {answer && (
          <div>
            <h2>Answer:</h2>
            <p>{answer}</p>
          </div>
        )}
      </div>
    </>
  );
}

export default App;
