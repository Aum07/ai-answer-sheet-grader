// Initialize Dropzone for file uploads
Dropzone.autoDiscover = false;

let referenceFile = null;
let studentFiles = [];
let progressInterval;

document.addEventListener('DOMContentLoaded', function() {
    // Reference file dropzone
    const referenceDropzone = new Dropzone("#referenceDropzone", {
        url: "/evaluate/",
        autoProcessQueue: false,
        maxFiles: 1,
        acceptedFiles: ".pdf",
        addRemoveLinks: true,
        dictDefaultMessage: "<i class='fas fa-file-pdf fa-2x mb-3'></i><br>Drop reference answer sheet here<br><span class='text-muted'>or click to upload</span>",
    });

    // Student files dropzone
    const studentDropzone = new Dropzone("#studentDropzone", {
        url: "/evaluate/",
        autoProcessQueue: false,
        maxFiles: 30,
        acceptedFiles: ".pdf",
        addRemoveLinks: true,
        dictDefaultMessage: "<i class='fas fa-file-pdf fa-2x mb-3'></i><br>Drop student answer sheets here<br><span class='text-muted'>or click to upload</span>",
    });

    // Handle reference file upload
    referenceDropzone.on("addedfile", function(file) {
        if (referenceDropzone.files.length > 1) {
            referenceDropzone.removeFile(referenceDropzone.files[0]);
        }
        referenceFile = file;
        updateSubmitButton();
        
        // Add animation to the file preview
        setTimeout(() => {
            file.previewElement.classList.add('animate__animated', 'animate__bounceIn');
        }, 100);
    });

    referenceDropzone.on("removedfile", function() {
        referenceFile = null;
        updateSubmitButton();
    });

    // Handle student files upload
    studentDropzone.on("addedfile", function(file) {
        studentFiles.push(file);
        updateSubmitButton();
        
        // Add animation to the file preview
        setTimeout(() => {
            file.previewElement.classList.add('animate__animated', 'animate__bounceIn');
        }, 100);
    });

    // Dropzone still fires "addedfile" for files over maxFiles; remove them so they are not uploaded
    studentDropzone.on("maxfilesexceeded", function(file) {
        studentDropzone.removeFile(file);
        alertOnce(`You can grade up to ${studentDropzone.options.maxFiles} student sheets at a time. Extra files were not added.`);
    });

    studentDropzone.on("removedfile", function(file) {
        const index = studentFiles.indexOf(file);
        if (index !== -1) {
            studentFiles.splice(index, 1);
        }
        updateSubmitButton();
    });

    // Enable/disable submit button based on file uploads
    function updateSubmitButton() {
        const submitBtn = document.getElementById('submitBtn');
        submitBtn.disabled = !referenceFile || studentFiles.length === 0;
        
        if (!referenceFile || studentFiles.length === 0) {
            submitBtn.classList.remove('btn-pulse');
        } else {
            submitBtn.classList.add('btn-pulse');
        }
    }

    // Demo: grade built-in sample sheets on the server (no uploads needed)
    document.getElementById('demoBtn').addEventListener('click', () => runEvaluation(() => fetch('/demo/', { method: 'POST' })));

    // Handle form submission
    document.getElementById('submitBtn').addEventListener('click', () => {
        const formData = new FormData();
        formData.append('reference_pdf', referenceFile);
        studentFiles.forEach(file => formData.append('student_pdfs', file));
        runEvaluation(() => fetch('/evaluate/', { method: 'POST', body: formData }));
    });
});

// Show the progress modal, run the request, then open the results page (or show the error)
async function runEvaluation(makeRequest) {
    const processingModal = new bootstrap.Modal(document.getElementById('processingModal'));
    processingModal.show();

    let progress = 0;
    const progressBar = document.getElementById('progressBar');
    progressInterval = setInterval(() => {
        if (progress < 90) {
            progress += (90 - progress) / 10;
            progressBar.style.width = `${progress}%`;
        }
    }, 500);

    try {
        const response = await makeRequest();
        clearInterval(progressInterval);
        progressBar.style.width = '100%';

        if (!response.ok) {
            let detail = `Status ${response.status}`;
            try { detail = (await response.json()).detail || detail; } catch (_) {}
            throw new Error(detail);
        }

        const data = await response.json();
        localStorage.setItem('evaluationResults', JSON.stringify(data));
        setTimeout(() => { window.location.href = '/static/results.html'; }, 500);
    } catch (error) {
        console.error('Error:', error);
        clearInterval(progressInterval);
        processingModal.hide();
        const errorModal = new bootstrap.Modal(document.getElementById('errorModal'));
        document.getElementById('errorMessage').textContent = error.message;
        errorModal.show();
    }
}

let _alerted = false;
function alertOnce(message) {
    if (_alerted) return;
    _alerted = true;
    alert(message);
    setTimeout(() => { _alerted = false; }, 2000);
}
