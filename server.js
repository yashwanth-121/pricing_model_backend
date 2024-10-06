const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const cors = require('cors');  // Import cors middleware
const { spawn } =  require('child_process');

// Create an instance of express
const app = express();

// Enable CORS for all routes
app.use(cors());

// Path to the file where 'sample.csv' will be stored (root directory)
const filePath = path.join(__dirname, 'sample.csv');

// Set up multer storage to always use 'sample.csv' as the file name in the root directory
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, __dirname);  // Root directory
  },
  filename: function (req, file, cb) {
    // Check if the file already exists
    if (fs.existsSync(filePath)) {
      // Delete the existing file
      fs.unlinkSync(filePath);
      console.log('Existing sample.csv deleted');
    }
    cb(null, 'sample.csv');  // Always set file name to 'sample.csv'
  }
});

// File filter to accept only .csv files
const fileFilter = (req, file, cb) => {
  if (file.mimetype === 'text/csv') {
    cb(null, true);
  } else {
    cb(new Error('Only .csv files are allowed!'), false);
  }
};

// Set up multer middleware
const upload = multer({
    storage: storage,
    fileFilter: (req, file, cb) => {
      // Allow any file type by always returning true
      cb(null, true);
    }
  });

// Route to handle file upload (single .csv file)
app.post('/upload', upload.single('sample'), (req, res) => {
  if (!req.file) {
    return res.status(400).send('No file uploaded or invalid file type. Please upload a .csv file.');
  }

  const childPython = spawn('python', ['test.py']);
  let finalData = "";
  childPython.stdout.on('data', (data) => {
   finalData =  data.toString()
  })

  childPython.stderr.on('data', (data) => {
    console.error('error', data.toString())
  })

  childPython.on('close', (code) => {
    console.log(`child process exited with code ${code}`);
    return res.status(200).send(finalData)
})

});

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
