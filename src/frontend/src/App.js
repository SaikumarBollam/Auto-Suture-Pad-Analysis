import React, { useState } from 'react';
import { Box, Container, Typography, CircularProgress } from '@mui/material';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const onDrop = async (acceptedFiles) => {
    setAnalyzing(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', acceptedFiles[0]);

    try {
      const response = await axios.post(`${API_URL}/api/v1/analyze`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResults(response.data.results);
    } catch (err) {
      setError(err.response?.data?.error || 'Error analyzing image');
    } finally {
      setAnalyzing(false);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png']
    },
    multiple: false
  });

  return (
    <Container maxWidth="md">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          Suture Analysis
        </Typography>

        <Box
          {...getRootProps()}
          sx={{
            border: '2px dashed #ccc',
            borderRadius: 2,
            p: 4,
            textAlign: 'center',
            cursor: 'pointer',
            '&:hover': {
              borderColor: 'primary.main'
            }
          }}
        >
          <input {...getInputProps()} />
          {analyzing ? (
            <CircularProgress />
          ) : isDragActive ? (
            <Typography>Drop the image here...</Typography>
          ) : (
            <Typography>
              Drag and drop an image here, or click to select
            </Typography>
          )}
        </Box>

        {error && (
          <Typography color="error" sx={{ mt: 2 }}>
            {error}
          </Typography>
        )}

        {results && (
          <Box sx={{ mt: 4 }}>
            <Typography variant="h6" gutterBottom>
              Analysis Results
            </Typography>
            {results.map((result, index) => (
              <Box key={index} sx={{ mb: 1 }}>
                <Typography>
                  Suture {index + 1}: Score {(result.score * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Position: ({result.box.map(n => n.toFixed(1)).join(', ')})
                </Typography>
              </Box>
            ))}
          </Box>
        )}
      </Box>
    </Container>
  );
}

export default App;