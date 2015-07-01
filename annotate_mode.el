;; define a var for your keymap, so that you can set it as local map
;; (meaning, active only when your mode is on)
(defvar annotate-mode-map nil "Keymap for annotate-mode")

;; definition for your keybinding and menu
(when (not annotate-mode-map) ; if it is not already defined

  ;; assign command to keys
  (setq annotate-mode-map (make-sparse-keymap))
  (define-key annotate-mode-map (kbd "q") 'annotate-skip-word)
  (define-key annotate-mode-map (kbd "w") 'annotate-mark-word))

(defun annotate-skip-word ()
  (interactive)
  (right-word)
  (right-word)
  (left-word))

(defun annotate-mark-word ()
  (interactive)
  (insert "//MARK\\\\")
  (right-word)
  (right-word)
  (left-word))



  
(defun annotate-mode ()
  "Major mode for annotating my HeadFi dumps"
  (interactive)
  (kill-all-local-variables)

  (setq major-mode 'annotate-mode)
  (setq mode-name "annotate") ; for display purposes in mode line
  (use-local-map annotate-mode-map)

  ;; … other code here

  (run-hooks 'annotate-mode-hook))

;; put your mode symbol into the list “features”, so that user can
;; call (require 'annotate-mode), which will load your code only when
;; needed
(provide 'annotate-mode)
