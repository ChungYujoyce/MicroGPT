// Function to enable editing of the clicked string
function editString(button) {
    // Get the parent <li> element
    var listItem = button.parentNode;

    // Get the <span> element containing the string
    var spanElement = listItem.querySelector("span");

    // Display input field with the current string value
    var stringValue = spanElement.textContent;
    var editInput = document.getElementById("editInput");
    editInput.value = stringValue;
    editInput.style.display = "block";
    editInput.focus();

    // Hide the <span> element and the "Revise" button
    spanElement.style.display = "none";
    button.style.display = "none";

    // Display the save button associated with the edited string
    var saveButton = listItem.querySelector(".saveButton");
    saveButton.style.display = "inline-block";
}

// Function to save the edited string
function saveEditedString(button, id) {
    event.preventDefault();

    // Get the parent <li> element
    var listItem = button.parentNode;

    // Get the edited string from the input field
    var editedString = document.getElementById("editInput").value;

    // Update the <span> element with the new value
    var spanElement = listItem.querySelector("span");
    spanElement.textContent = editedString;

    // Hide the input field and save button
    var editInput = document.getElementById("editInput");
    editInput.style.display = "none";
    button.style.display = "none";

    // Show the <span> element and "Revise" button
    spanElement.style.display = "block";
    var reviseButton = listItem.querySelector(".reviseButton");
    reviseButton.style.display = "inline-block";

    // id 
    // editedString
    var form = document.getElementById("dataForm");
    var formData = new FormData();
    formData.append("editInput", true);
    formData.append("id", id);
    formData.append("revise_result", editedString);

    // Fetch API to submit the form data
    fetch(form.action, {
        method: "PUT",
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error("Network response was not ok");
        }
        // Handle success response if needed
        console.log("Form submitted successfully");
    })
    .catch(error => {
        // Handle error if needed
        console.error("Error submitting form:", error.message);
    });


//         var input = document.createElement("input");
//         input.type = "hidden";
//         input.name = "id";
//         input.value = id;
//         form.appendChild(input);

//         var input = document.createElement("input");
//         input.type = "hidden";
//         input.name = "revise_result";
//         input.value = editedString;
//         form.appendChild(input);

//         form.submit();
}

function deleteString(button, id, url) {
    event.preventDefault();

    // Get the parent <li> element
    var listItem = button.parentNode.parentNode;
    listItem.parentNode.removeChild(listItem);

    // id 
    // editedString
    var formData = new FormData();
    formData.append("deleteInput", true);
    formData.append("id", id);

    // Fetch API to submit the form data
    fetch(url, {
        method: "DELETE",
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error("Network response was not ok");
        }
        // Handle success response if needed
        console.log("Form submitted successfully");
    })
    .catch(error => {
        // Handle error if needed
        console.error("Error submitting form:", error.message);
    });
}

function openFileSelection() {
  const fileInput = document.getElementById("fileInput");
  fileInput.accept = ".csv, .pdf, .txt, .doc";
  fileInput.click();
  fileInput.addEventListener("change", handleFileSelection);
}

function handleFileSelection(event) {
  // You can perform some checks on the files here if you want
  // Open the modal after file selection
  const uploadModal = new bootstrap.Modal(
    document.getElementById("uploadModal"),
  );
  uploadModal.show();
}

function submitPromptForm() {
  // Show the modal
  $("#responseModal").modal("show");

  // Submit the form after a short delay to allow the modal to open
  setTimeout(function () {
    document.getElementById("promptForm").submit();
  }, 5);
}

function submitForm(action) {
  var form = document.getElementById("uploadForm");

  var input = document.createElement("input");
  input.type = "hidden";
  input.name = "action";
  input.value = action;

  form.appendChild(input);

  // After the form is submitted, close the current modal and open the new one.
  $("#uploadModal").on("hidden.bs.modal", function () {
    $("#ingesting-modal").modal("show");
  });

  if (action == "add" || action == "reset") {
    $("#uploadModal").modal("hide");
  }

  form.submit();
}
