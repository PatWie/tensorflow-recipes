from subprocess import Popen, PIPE


class GTPbridge(object):
    """A communication pipe to GTP application.
    """
    def __init__(self, name, pipe_args, verbose=True):
        """Create a pipe to GTP app.

        Args:
            name (str): identifier for terminal output
            pipe_args ([str]): list of arguments
            verbose (bool, optional): set verbose output (terminal)
        """
        self.name = name
        self.verbose = verbose
        self.subprocess = Popen(pipe_args, stdin=PIPE, stdout=PIPE)
        print("creating pipe for {}".format(name))

    def send(self, cmd):
        """Send command to GTP app.

        Args:
            cmd (str): Command (has to end with \n)

        Returns:
            str: result of command (you might want to clean the output)
        """
        if self.verbose:
            print("[{}] receives '{}'".format(self.name, cmd.replace('\n', '')))
        self.subprocess.stdin.write(cmd)
        result = ""
        while True:
            data = self.subprocess.stdout.readline()
            if not data.strip():
                break
            result += data
        if self.verbose:
            print("[{}] returns {}".format(self.name, result))
        return result

    def close(self):
        """Proper closing of pipe.
        """
        print("closing pipe to {}".format(self.name))
        self.subprocess.communicate("quit\n")
