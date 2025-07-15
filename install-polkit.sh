sudo tee /etc/polkit-1/rules.d/49-allow-systemctl-and-shutdown.rules > /dev/null <<'EOL'
polkit.addRule(function(action, subject) {
    if (
        subject.user == "transformirror1" &&
        (
            action.id == "org.freedesktop.systemd1.manage-units" ||
            action.id == "org.freedesktop.login1.power-off"
        )
    ) {
        return polkit.Result.YES;
    }
});
EOL